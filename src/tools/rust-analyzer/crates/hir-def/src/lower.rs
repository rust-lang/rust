//! Context for lowering paths.
use std::cell::{OnceCell, RefCell};

use hir_expand::{
    span_map::{SpanMap, SpanMapRef},
    AstId, HirFileId, InFile,
};
use span::{AstIdMap, AstIdNode};
use stdx::thin_vec::ThinVec;
use syntax::ast;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    path::Path,
    type_ref::{TypeBound, TypePtr, TypeRef, TypeRefId, TypesMap, TypesSourceMap},
};

pub struct LowerCtx<'a> {
    pub db: &'a dyn DefDatabase,
    file_id: HirFileId,
    span_map: OnceCell<SpanMap>,
    ast_id_map: OnceCell<Arc<AstIdMap>>,
    impl_trait_bounds: RefCell<Vec<ThinVec<TypeBound>>>,
    // Prevent nested impl traits like `impl Foo<impl Bar>`.
    outer_impl_trait: RefCell<bool>,
    types_map: RefCell<(&'a mut TypesMap, &'a mut TypesSourceMap)>,
}

pub(crate) struct OuterImplTraitGuard<'a, 'b> {
    ctx: &'a LowerCtx<'b>,
    old: bool,
}

impl<'a, 'b> OuterImplTraitGuard<'a, 'b> {
    fn new(ctx: &'a LowerCtx<'b>, impl_trait: bool) -> Self {
        let old = ctx.outer_impl_trait.replace(impl_trait);
        Self { ctx, old }
    }
}

impl Drop for OuterImplTraitGuard<'_, '_> {
    fn drop(&mut self) {
        self.ctx.outer_impl_trait.replace(self.old);
    }
}

impl<'a> LowerCtx<'a> {
    pub fn new(
        db: &'a dyn DefDatabase,
        file_id: HirFileId,
        types_map: &'a mut TypesMap,
        types_source_map: &'a mut TypesSourceMap,
    ) -> Self {
        LowerCtx {
            db,
            file_id,
            span_map: OnceCell::new(),
            ast_id_map: OnceCell::new(),
            impl_trait_bounds: RefCell::new(Vec::new()),
            outer_impl_trait: RefCell::default(),
            types_map: RefCell::new((types_map, types_source_map)),
        }
    }

    pub fn with_span_map_cell(
        db: &'a dyn DefDatabase,
        file_id: HirFileId,
        span_map: OnceCell<SpanMap>,
        types_map: &'a mut TypesMap,
        types_source_map: &'a mut TypesSourceMap,
    ) -> Self {
        LowerCtx {
            db,
            file_id,
            span_map,
            ast_id_map: OnceCell::new(),
            impl_trait_bounds: RefCell::new(Vec::new()),
            outer_impl_trait: RefCell::default(),
            types_map: RefCell::new((types_map, types_source_map)),
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

    pub fn update_impl_traits_bounds(&self, bounds: ThinVec<TypeBound>) {
        self.impl_trait_bounds.borrow_mut().push(bounds);
    }

    pub fn take_impl_traits_bounds(&self) -> Vec<ThinVec<TypeBound>> {
        self.impl_trait_bounds.take()
    }

    pub(crate) fn outer_impl_trait(&self) -> bool {
        *self.outer_impl_trait.borrow()
    }

    pub(crate) fn outer_impl_trait_scope<'b>(
        &'b self,
        impl_trait: bool,
    ) -> OuterImplTraitGuard<'b, 'a> {
        OuterImplTraitGuard::new(self, impl_trait)
    }

    pub(crate) fn alloc_type_ref(&self, type_ref: TypeRef, node: TypePtr) -> TypeRefId {
        let mut types_map = self.types_map.borrow_mut();
        let (types_map, types_source_map) = &mut *types_map;
        let id = types_map.types.alloc(type_ref);
        types_source_map.types_map_back.insert(id, InFile::new(self.file_id, node));
        id
    }

    pub(crate) fn alloc_type_ref_desugared(&self, type_ref: TypeRef) -> TypeRefId {
        self.types_map.borrow_mut().0.types.alloc(type_ref)
    }

    pub(crate) fn alloc_error_type(&self) -> TypeRefId {
        self.types_map.borrow_mut().0.types.alloc(TypeRef::Error)
    }

    // FIXME: If we alloc while holding this, well... Bad Things will happen. Need to change this
    // to use proper mutability instead of interior mutability.
    pub(crate) fn types_map(&self) -> std::cell::Ref<'_, TypesMap> {
        std::cell::Ref::map(self.types_map.borrow(), |it| &*it.0)
    }
}
