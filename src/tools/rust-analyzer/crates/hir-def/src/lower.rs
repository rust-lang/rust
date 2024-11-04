//! Context for lowering paths.
use std::{cell::OnceCell, mem};

use hir_expand::{span_map::SpanMap, AstId, HirFileId, InFile};
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
    impl_trait_bounds: Vec<ThinVec<TypeBound>>,
    // Prevent nested impl traits like `impl Foo<impl Bar>`.
    outer_impl_trait: bool,
    types_map: &'a mut TypesMap,
    types_source_map: &'a mut TypesSourceMap,
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
            impl_trait_bounds: Vec::new(),
            outer_impl_trait: false,
            types_map,
            types_source_map,
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
            impl_trait_bounds: Vec::new(),
            outer_impl_trait: false,
            types_map,
            types_source_map,
        }
    }

    pub(crate) fn span_map(&self) -> &SpanMap {
        self.span_map.get_or_init(|| self.db.span_map(self.file_id))
    }

    pub(crate) fn lower_path(&mut self, ast: ast::Path) -> Option<Path> {
        Path::from_src(self, ast)
    }

    pub(crate) fn ast_id<N: AstIdNode>(&self, item: &N) -> AstId<N> {
        InFile::new(
            self.file_id,
            self.ast_id_map.get_or_init(|| self.db.ast_id_map(self.file_id)).ast_id(item),
        )
    }

    pub fn update_impl_traits_bounds_from_type_ref(&mut self, type_ref: TypeRefId) {
        TypeRef::walk(type_ref, self.types_map, &mut |tr| {
            if let TypeRef::ImplTrait(bounds) = tr {
                self.impl_trait_bounds.push(bounds.clone());
            }
        });
    }

    pub fn take_impl_traits_bounds(&mut self) -> Vec<ThinVec<TypeBound>> {
        mem::take(&mut self.impl_trait_bounds)
    }

    pub(crate) fn outer_impl_trait(&self) -> bool {
        self.outer_impl_trait
    }

    pub(crate) fn with_outer_impl_trait_scope<R>(
        &mut self,
        impl_trait: bool,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old = mem::replace(&mut self.outer_impl_trait, impl_trait);
        let result = f(self);
        self.outer_impl_trait = old;
        result
    }

    pub(crate) fn alloc_type_ref(&mut self, type_ref: TypeRef, node: TypePtr) -> TypeRefId {
        let id = self.types_map.types.alloc(type_ref);
        self.types_source_map.types_map_back.insert(id, InFile::new(self.file_id, node));
        id
    }

    pub(crate) fn alloc_type_ref_desugared(&mut self, type_ref: TypeRef) -> TypeRefId {
        self.types_map.types.alloc(type_ref)
    }

    pub(crate) fn alloc_error_type(&mut self) -> TypeRefId {
        self.types_map.types.alloc(TypeRef::Error)
    }
}
