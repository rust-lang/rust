use rustc_macros::SessionDiagnostic;
use rustc_span::{Span, Symbol};

#[derive(SessionDiagnostic)]
#[diag(interface::ferris_identifier)]
pub struct FerrisIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    #[suggestion(code = "ferris", applicability = "maybe-incorrect")]
    pub first_span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(interface::emoji_identifier)]
pub struct EmojiIdentifier {
    #[primary_span]
    pub spans: Vec<Span>,
    pub ident: Symbol,
}
