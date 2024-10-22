//! Context for lowering paths.
use std::cell::{OnceCell, RefCell};

use hir_expand::{
    span_map::{SpanMap, SpanMapRef},
    AstId, HirFileId, InFile,
};
use intern::Interned;
use span::{AstIdMap, AstIdNode};
use syntax::ast;
use triomphe::Arc;

use crate::{db::DefDatabase, path::Path, type_ref::TypeBound};

pub struct LowerCtx<'a> {
    pub db: &'a dyn DefDatabase,
    file_id: HirFileId,
    span_map: OnceCell<SpanMap>,
    ast_id_map: OnceCell<Arc<AstIdMap>>,
    impl_trait_bounds: RefCell<Vec<Vec<Interned<TypeBound>>>>,
    // Prevent nested impl traits like `impl Foo<impl Bar>`.
    outer_impl_trait: RefCell<bool>,
}

pub(crate) struct OuterImplTraitGuard<'a> {
    ctx: &'a LowerCtx<'a>,
    old: bool,
}

impl<'a> OuterImplTraitGuard<'a> {
    fn new(ctx: &'a LowerCtx<'a>, impl_trait: bool) -> Self {
        let old = ctx.outer_impl_trait.replace(impl_trait);
        Self { ctx, old }
    }
}

impl Drop for OuterImplTraitGuard<'_> {
    fn drop(&mut self) {
        self.ctx.outer_impl_trait.replace(self.old);
    }
}

impl<'a> LowerCtx<'a> {
    pub fn new(db: &'a dyn DefDatabase, file_id: HirFileId) -> Self {
        LowerCtx {
            db,
            file_id,
            span_map: OnceCell::new(),
            ast_id_map: OnceCell::new(),
            impl_trait_bounds: RefCell::new(Vec::new()),
            outer_impl_trait: RefCell::default(),
        }
    }

    pub fn with_span_map_cell(
        db: &'a dyn DefDatabase,
        file_id: HirFileId,
        span_map: OnceCell<SpanMap>,
    ) -> Self {
        LowerCtx {
            db,
            file_id,
            span_map,
            ast_id_map: OnceCell::new(),
            impl_trait_bounds: RefCell::new(Vec::new()),
            outer_impl_trait: RefCell::default(),
        }
    }

    pub(crate) fn span_map(&self) -> SpanMapRef<'_> {
        self.span_map.get_or_init(|| self.db.span_map(self.file_id)).as_ref()
    }

    pub(crate) fn lower_path(&self, ast: ast::Path) -> Option<Path> {
        Path::from_src(self, ast)
    }

    pub(crate) fn ast_id<N: AstIdNode>(&self, item: &N) -> AstId<N> {
        InFile::new(
            self.file_id,
            self.ast_id_map.get_or_init(|| self.db.ast_id_map(self.file_id)).ast_id(item),
        )
    }

    pub fn update_impl_traits_bounds(&self, bounds: Vec<Interned<TypeBound>>) {
        self.impl_trait_bounds.borrow_mut().push(bounds);
    }

    pub fn take_impl_traits_bounds(&self) -> Vec<Vec<Interned<TypeBound>>> {
        self.impl_trait_bounds.take()
    }

    pub(crate) fn outer_impl_trait(&self) -> bool {
        *self.outer_impl_trait.borrow()
    }

    pub(crate) fn outer_impl_trait_scope(&'a self, impl_trait: bool) -> OuterImplTraitGuard<'a> {
        OuterImplTraitGuard::new(self, impl_trait)
    }
}
