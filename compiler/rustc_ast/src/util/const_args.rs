use std::ops::ControlFlow;

use crate::visit::{Visitor, walk_anon_const};
use crate::{DUMMY_NODE_ID, Expr, ExprKind, Path};

impl Expr {
    // FIXME: update docs
    /// Could this expr be either `N`, or `{ N }`, where `N` is a const parameter.
    ///
    /// If this is not the case, name resolution does not resolve `N` when using
    /// `min_const_generics` as more complex expressions are not supported.
    ///
    /// Does not ensure that the path resolves to a const param, the caller should check this.
    /// This also does not consider macros, so it's only correct after macro-expansion.
    pub fn is_potential_trivial_const_arg(&self, allow_mgca_arg: bool) -> bool {
        let this = self.maybe_unwrap_block();
        if allow_mgca_arg {
            MGCATrivialConstArgVisitor::new().visit_expr(this).is_continue()
        } else {
            if let ExprKind::Path(None, path) = &this.kind
                && path.is_potential_trivial_const_arg(allow_mgca_arg)
            {
                true
            } else {
                false
            }
        }
    }
}

impl Path {
    // FIXME: add docs
    #[tracing::instrument(level = "debug", ret)]
    pub fn is_potential_trivial_const_arg(&self, allow_mgca_arg: bool) -> bool {
        if allow_mgca_arg {
            MGCATrivialConstArgVisitor::new().visit_path(self, DUMMY_NODE_ID).is_continue()
        } else {
            self.segments.len() == 1 && self.segments.iter().all(|seg| seg.args.is_none())
        }
    }
}

pub(crate) struct MGCATrivialConstArgVisitor {}

impl MGCATrivialConstArgVisitor {
    pub(crate) fn new() -> Self {
        Self {}
    }
}

impl<'ast> Visitor<'ast> for MGCATrivialConstArgVisitor {
    type Result = ControlFlow<()>;

    fn visit_anon_const(&mut self, c: &'ast crate::AnonConst) -> Self::Result {
        let expr = c.value.maybe_unwrap_block();
        match &expr.kind {
            ExprKind::Path(_, _) => {}
            _ => return ControlFlow::Break(()),
        }
        walk_anon_const(self, c)
    }
}
