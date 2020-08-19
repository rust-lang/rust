use std::collections::BTreeSet;
use std::cmp;
use rustc_span::Span;
use rustc_span::symbol::Symbol;

#[derive(Eq)]
pub struct BindingError {
    pub(super) name: Symbol,
    pub(super) origin: BTreeSet<Span>,
    pub(super) target: BTreeSet<Span>,
    pub(super) could_be_path: bool,
}

impl PartialOrd for BindingError {
    fn partial_cmp(&self, other: &BindingError) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BindingError {
    fn eq(&self, other: &BindingError) -> bool {
        self.name == other.name
    }
}

impl Ord for BindingError {
    fn cmp(&self, other: &BindingError) -> cmp::Ordering {
        self.name.cmp(&other.name)
    }
}