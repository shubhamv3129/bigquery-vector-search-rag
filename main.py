"""
Main entry point for the Enterprise GCP RAG Demo application.
"""
from feature_rag_pipeline import render_rag_demo_app


def example_rag_demo() -> None:
    """
    Example function launching the Enterprise RAG Streamlit UI.
    """
    render_rag_demo_app()


def main() -> None:
    """
    Main entry point.
    """
    example_rag_demo()


if __name__ == "__main__":
    main()

