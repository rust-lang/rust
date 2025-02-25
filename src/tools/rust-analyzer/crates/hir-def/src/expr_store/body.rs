//! Defines `Body`: a lowered representation of functions, statics and
//! consts.
use std::ops;

use hir_expand::{InFile, Lookup};
use la_arena::{Idx, RawIdx};
use span::Edition;
use syntax::ast;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    expander::Expander,
    expr_store::{lower, pretty, ExpressionStore, ExpressionStoreSourceMap, SelfParamPtr},
    hir::{BindingId, ExprId, PatId},
    item_tree::AttrOwner,
    src::HasSource,
    DefWithBodyId, HasModule,
};

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    pub store: ExpressionStore,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    pub params: Box<[PatId]>,
    pub self_param: Option<BindingId>,
    /// The `ExprId` of the actual body expression.
    pub body_expr: ExprId,
}

impl ops::Deref for Body {
    type Target = ExpressionStore;

    fn deref(&self) -> &Self::Target {
        &self.store
    }
}

/// An item body together with the mapping from syntax nodes to HIR expression
/// IDs. This is needed to go from e.g. a position in a file to the HIR
/// expression containing it; but for type inference etc., we want to operate on
/// a structure that is agnostic to the actual positions of expressions in the
/// file, so that we don't recompute types whenever some whitespace is typed.
///
/// One complication here is that, due to macro expansion, a single `Body` might
/// be spread across several files. So, for each ExprId and PatId, we record
/// both the HirFileId and the position inside the file. However, we only store
/// AST -> ExprId mapping for non-macro files, as it is not clear how to handle
/// this properly for macros.
#[derive(Default, Debug, Eq, PartialEq)]
pub struct BodySourceMap {
    pub self_param: Option<InFile<SelfParamPtr>>,
    pub store: ExpressionStoreSourceMap,
}

impl ops::Deref for BodySourceMap {
    type Target = ExpressionStoreSourceMap;

    fn deref(&self) -> &Self::Target {
        &self.store
    }
}

impl Body {
    pub(crate) fn body_with_source_map_query(
        db: &dyn DefDatabase,
        def: DefWithBodyId,
    ) -> (Arc<Body>, Arc<BodySourceMap>) {
        let _p = tracing::info_span!("body_with_source_map_query").entered();
        let mut params = None;

        let mut is_async_fn = false;
        let InFile { file_id, value: body } = {
            match def {
                DefWithBodyId::FunctionId(f) => {
                    let data = db.function_data(f);
                    let f = f.lookup(db);
                    let src = f.source(db);
                    params = src.value.param_list().map(move |param_list| {
                        let item_tree = f.id.item_tree(db);
                        let func = &item_tree[f.id.value];
                        let krate = f.container.module(db).krate;
                        let crate_graph = db.crate_graph();
                        (
                            param_list,
                            (0..func.params.len()).map(move |idx| {
                                item_tree
                                    .attrs(
                                        db,
                                        krate,
                                        AttrOwner::Param(
                                            f.id.value,
                                            Idx::from_raw(RawIdx::from(idx as u32)),
                                        ),
                                    )
                                    .is_cfg_enabled(&crate_graph[krate].cfg_options)
                            }),
                        )
                    });
                    is_async_fn = data.is_async();
                    src.map(|it| it.body().map(ast::Expr::from))
                }
                DefWithBodyId::ConstId(c) => {
                    let c = c.lookup(db);
                    let src = c.source(db);
                    src.map(|it| it.body())
                }
                DefWithBodyId::StaticId(s) => {
                    let s = s.lookup(db);
                    let src = s.source(db);
                    src.map(|it| it.body())
                }
                DefWithBodyId::VariantId(v) => {
                    let s = v.lookup(db);
                    let src = s.source(db);
                    src.map(|it| it.expr())
                }
                DefWithBodyId::InTypeConstId(c) => c.lookup(db).id.map(|_| c.source(db).expr()),
            }
        };
        let module = def.module(db);
        let expander = Expander::new(db, file_id, module);
        let (body, mut source_map) =
            lower::lower_body(db, def, expander, params, body, module.krate, is_async_fn);
        source_map.store.shrink_to_fit();

        (Arc::new(body), Arc::new(source_map))
    }

    pub(crate) fn body_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<Body> {
        db.body_with_source_map(def).0
    }

    pub fn pretty_print(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        edition: Edition,
    ) -> String {
        pretty::print_body_hir(db, self, owner, edition)
    }

    pub fn pretty_print_expr(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        expr: ExprId,
        edition: Edition,
    ) -> String {
        pretty::print_expr_hir(db, self, owner, expr, edition)
    }

    pub fn pretty_print_pat(
        &self,
        db: &dyn DefDatabase,
        owner: DefWithBodyId,
        pat: PatId,
        oneline: bool,
        edition: Edition,
    ) -> String {
        pretty::print_pat_hir(db, self, owner, pat, oneline, edition)
    }
}

impl BodySourceMap {
    pub fn self_param_syntax(&self) -> Option<InFile<SelfParamPtr>> {
        self.self_param
    }
}
