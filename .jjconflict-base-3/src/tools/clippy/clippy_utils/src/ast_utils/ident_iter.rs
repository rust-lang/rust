use core::iter::FusedIterator;
use rustc_ast::visit::{Visitor, walk_attribute, walk_expr};
use rustc_ast::{Attribute, Expr};
use rustc_span::symbol::Ident;

pub struct IdentIter(std::vec::IntoIter<Ident>);

impl Iterator for IdentIter {
    type Item = Ident;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl FusedIterator for IdentIter {}

impl From<&Expr> for IdentIter {
    fn from(expr: &Expr) -> Self {
        let mut visitor = IdentCollector::default();

        walk_expr(&mut visitor, expr);

        IdentIter(visitor.0.into_iter())
    }
}

impl From<&Attribute> for IdentIter {
    fn from(attr: &Attribute) -> Self {
        let mut visitor = IdentCollector::default();

        walk_attribute(&mut visitor, attr);

        IdentIter(visitor.0.into_iter())
    }
}

#[derive(Default)]
struct IdentCollector(Vec<Ident>);

impl Visitor<'_> for IdentCollector {
    fn visit_ident(&mut self, ident: &Ident) {
        self.0.push(*ident);
    }
}
