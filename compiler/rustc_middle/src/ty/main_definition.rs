use rustc_span::Span;

use rustc_ast as ast;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;

#[derive(Clone, Copy, Debug)]
pub struct MainDefinition {
    pub res: Res<ast::NodeId>,
    pub is_import: bool,
    pub span: Span,
}

impl MainDefinition {
    pub fn opt_fn_def_id(self) -> Option<DefId> {
        if let Res::Def(DefKind::Fn, def_id) = self.res { Some(def_id) } else { None }
    }
}
