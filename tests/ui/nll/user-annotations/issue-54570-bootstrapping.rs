//@ check-pass

// This test is reduced from a scenario pnkfelix encountered while
// bootstrapping the compiler.

#[derive(Copy, Clone)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

pub type Variant = Spanned<VariantKind>;
// #[derive(Clone)] pub struct Variant { pub node: VariantKind, pub span: Span, }

#[derive(Clone)]
pub struct VariantKind { }

#[derive(Copy, Clone)]
pub struct Span;

pub fn variant_to_span(variant: Variant) {
    match variant {
        Variant {
            span: _span,
            ..
        } => { }
    };
}

fn main() { }
