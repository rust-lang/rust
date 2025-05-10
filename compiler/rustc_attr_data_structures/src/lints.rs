// pub type HirDelayedLint = (
//     &'static Lint,
//     HirId,
//     Span,
//     Box<dyn DynSend + for<'a, 'b> FnOnce(&'b mut Diag<'a, ()>) + 'static>,
// );

use rustc_span::Span;

pub enum AttributeLintKind {
    UnusedDuplicate { unused: Span, used: Span, warning: bool },
}

pub struct AttributeLint<Id> {
    pub id: Id,
    pub kind: AttributeLintKind,
}
