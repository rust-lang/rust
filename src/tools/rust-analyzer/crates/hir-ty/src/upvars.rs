//! A simple query to collect tall locals (upvars) a closure use.

use hir_def::{
    DefWithBodyId, ExpressionStoreOwnerId, GenericDefId, VariantId,
    expr_store::{ExpressionStore, path::Path},
    hir::{BindingId, Expr, ExprId, ExprOrPatId},
    resolver::{HasResolver, Resolver, ValueNs},
};
use hir_expand::mod_path::PathKind;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::db::HirDatabase;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
// Kept sorted.
pub struct Upvars(Box<[BindingId]>);

impl Upvars {
    fn new(upvars: &FxHashSet<BindingId>) -> Upvars {
        let mut upvars = upvars.iter().copied().collect::<Box<[_]>>();
        upvars.sort_unstable();
        Upvars(upvars)
    }

    #[inline]
    pub fn contains(&self, local: BindingId) -> bool {
        self.0.binary_search(&local).is_ok()
    }

    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = BindingId> {
        self.0.iter().copied()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn as_ref(&self) -> UpvarsRef<'_> {
        UpvarsRef(&self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
// Kept sorted.
pub struct UpvarsRef<'db>(&'db [BindingId]);

impl UpvarsRef<'_> {
    #[inline]
    pub fn contains(self, local: BindingId) -> bool {
        self.0.binary_search(&local).is_ok()
    }

    #[inline]
    pub fn iter(self) -> impl ExactSizeIterator<Item = BindingId> {
        self.0.iter().copied()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub const fn empty() -> Self {
        UpvarsRef(&[])
    }
}

/// Returns a map from `Expr::Closure` to its upvars.
pub fn upvars_mentioned(
    db: &dyn HirDatabase,
    owner: ExpressionStoreOwnerId,
) -> Option<&FxHashMap<ExprId, Upvars>> {
    return match owner {
        ExpressionStoreOwnerId::Signature(owner) => signature_upvars_mentioned(db, owner),
        ExpressionStoreOwnerId::Body(owner) => body_upvars_mentioned(db, owner),
        ExpressionStoreOwnerId::VariantFields(owner) => variant_fields_upvars_mentioned(db, owner),
    };

    #[salsa::tracked(returns(as_deref))]
    pub fn signature_upvars_mentioned(
        db: &dyn HirDatabase,
        owner: GenericDefId,
    ) -> Option<Box<FxHashMap<ExprId, Upvars>>> {
        upvars_mentioned_impl(db, owner.into())
    }

    #[salsa::tracked(returns(as_deref))]
    pub fn body_upvars_mentioned(
        db: &dyn HirDatabase,
        owner: DefWithBodyId,
    ) -> Option<Box<FxHashMap<ExprId, Upvars>>> {
        upvars_mentioned_impl(db, owner.into())
    }

    #[salsa::tracked(returns(as_deref))]
    pub fn variant_fields_upvars_mentioned(
        db: &dyn HirDatabase,
        owner: VariantId,
    ) -> Option<Box<FxHashMap<ExprId, Upvars>>> {
        upvars_mentioned_impl(db, owner.into())
    }
}

pub fn upvars_mentioned_impl(
    db: &dyn HirDatabase,
    owner: ExpressionStoreOwnerId,
) -> Option<Box<FxHashMap<ExprId, Upvars>>> {
    let store = ExpressionStore::of(db, owner);
    store.expr_roots().next()?;
    let mut resolver = owner.resolver(db);
    let mut result = FxHashMap::default();
    for root_expr in store.expr_roots() {
        handle_expr_outside_closure(db, &mut resolver, owner, store, root_expr, &mut result);
    }
    return if result.is_empty() {
        None
    } else {
        result.shrink_to_fit();
        Some(Box::new(result))
    };

    fn handle_expr_outside_closure<'db>(
        db: &'db dyn HirDatabase,
        resolver: &mut Resolver<'db>,
        owner: ExpressionStoreOwnerId,
        body: &ExpressionStore,
        expr: ExprId,
        closures_map: &mut FxHashMap<ExprId, Upvars>,
    ) {
        match &body[expr] {
            &Expr::Closure { body: body_expr, .. } => {
                let mut upvars = FxHashSet::default();
                handle_expr_inside_closure(
                    db,
                    resolver,
                    owner,
                    body,
                    expr,
                    body_expr,
                    &mut upvars,
                    closures_map,
                );
                if !upvars.is_empty() {
                    closures_map.insert(expr, Upvars::new(&upvars));
                }
            }
            _ => body.walk_child_exprs(expr, |expr| {
                handle_expr_outside_closure(db, resolver, owner, body, expr, closures_map)
            }),
        }
    }

    fn handle_expr_inside_closure<'db>(
        db: &'db dyn HirDatabase,
        resolver: &mut Resolver<'db>,
        owner: ExpressionStoreOwnerId,
        body: &ExpressionStore,
        current_closure: ExprId,
        expr: ExprId,
        upvars: &mut FxHashSet<BindingId>,
        closures_map: &mut FxHashMap<ExprId, Upvars>,
    ) {
        match &body[expr] {
            Expr::Path(path) => {
                resolve_maybe_upvar(
                    db,
                    resolver,
                    owner,
                    body,
                    current_closure,
                    expr,
                    expr.into(),
                    upvars,
                    path,
                );
            }
            &Expr::Closure { body: body_expr, .. } => {
                let mut closure_upvars = FxHashSet::default();
                handle_expr_inside_closure(
                    db,
                    resolver,
                    owner,
                    body,
                    expr,
                    body_expr,
                    &mut closure_upvars,
                    closures_map,
                );
                if !closure_upvars.is_empty() {
                    closures_map.insert(expr, Upvars::new(&closure_upvars));
                    // All nested closure's upvars are also upvars of the parent closure.
                    upvars.extend(
                        closure_upvars
                            .iter()
                            .copied()
                            .filter(|local| body.binding_owner(*local) != Some(current_closure)),
                    );
                }
                return;
            }
            _ => {}
        }
        body.walk_child_exprs(expr, |expr| {
            handle_expr_inside_closure(
                db,
                resolver,
                owner,
                body,
                current_closure,
                expr,
                upvars,
                closures_map,
            )
        });
    }
}

fn resolve_maybe_upvar<'db>(
    db: &'db dyn HirDatabase,
    resolver: &mut Resolver<'db>,
    owner: ExpressionStoreOwnerId,
    body: &ExpressionStore,
    current_closure: ExprId,
    expr: ExprId,
    id: ExprOrPatId,
    upvars: &mut FxHashSet<BindingId>,
    path: &Path,
) {
    if let Path::BarePath(mod_path) = path
        && matches!(mod_path.kind, PathKind::Plain | PathKind::SELF)
        // `self` is length zero.
        && mod_path.segments().len() <= 1
    {
        // Could be a variable.
        let guard = resolver.update_to_inner_scope(db, owner, expr);
        let resolution =
            resolver.resolve_path_in_value_ns_fully(db, path, body.expr_or_pat_path_hygiene(id));
        if let Some(ValueNs::LocalBinding(local)) = resolution
            && body.binding_owner(local) != Some(current_closure)
        {
            upvars.insert(local);
        }
        resolver.reset_to_guard(guard);
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{Expect, expect};
    use hir_def::{
        AssocItemId, DefWithBodyId, ModuleDefId, expr_store::Body, nameres::crate_def_map,
    };
    use itertools::Itertools;
    use span::Edition;
    use test_fixture::WithFixture;

    use crate::{test_db::TestDB, upvars::upvars_mentioned};

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expectation: Expect) {
        let db = TestDB::with_files(ra_fixture);
        crate::attach_db(&db, || {
            let def_map = crate_def_map(&db, db.test_crate());
            let func = def_map
                .modules()
                .flat_map(|(_, module)| module.scope.declarations())
                .filter_map(|decl| match decl {
                    ModuleDefId::FunctionId(func) => Some(func),
                    _ => None,
                })
                .chain(def_map.modules().flat_map(|(_, module)| {
                    module.scope.impls().flat_map(|impl_| &*impl_.impl_items(&db).items).filter_map(
                        |&(_, item)| match item {
                            AssocItemId::FunctionId(it) => Some(it),
                            _ => None,
                        },
                    )
                }))
                .exactly_one()
                .unwrap_or_else(|_| panic!("expected one function"));
            let (body, source_map) = Body::with_source_map(&db, func.into());
            let Some(upvars) = upvars_mentioned(&db, DefWithBodyId::from(func).into()) else {
                expectation.assert_eq("");
                return;
            };
            let mut closures = Vec::new();
            for (&closure, upvars) in upvars {
                let closure_range = source_map.expr_syntax(closure).unwrap().value.text_range();
                let upvars = upvars
                    .iter()
                    .map(|local| body[local].name.display(&db, Edition::CURRENT))
                    .join(", ");
                closures.push((closure_range, upvars));
            }
            closures.sort_unstable_by_key(|(range, _)| (range.start(), range.end()));
            let closures = closures
                .into_iter()
                .map(|(range, upvars)| format!("{range:?}: {upvars}"))
                .join("\n");
            expectation.assert_eq(&closures);
        });
    }

    #[test]
    fn simple() {
        check(
            r#"
struct foo;
fn foo(param: i32) {
    let local = "boo";
    || { param; foo };
    || local;
    || { param; local; param; local; };
    || 0xDEAFBEAF;
}
        "#,
            expect![[r#"
                60..77: param
                83..91: local
                97..131: param, local"#]],
        );
    }

    #[test]
    fn nested() {
        check(
            r#"
fn foo() {
    let (a, b);
    || {
        || a;
        || b;
    };
}
        "#,
            expect![[r#"
                31..69: a, b
                44..48: a
                58..62: b"#]],
        );
    }

    #[test]
    fn closure_var() {
        check(
            r#"
fn foo() {
    let upvar = 1;
    |closure_param: i32| {
        let closure_local = closure_param;
        closure_local + upvar
    };
}
        "#,
            expect!["34..135: upvar"],
        );
    }

    #[test]
    fn closure_var_nested() {
        check(
            r#"
fn foo() {
    let a = 1;
    |b: i32| {
        || {
            let c = 123;
            a + b + c
        }
    };
}
        "#,
            expect![[r#"
                30..116: a
                49..110: a, b"#]],
        );
    }

    #[test]
    fn self_upvar() {
        check(
            r#"
struct Foo(i32);
impl Foo {
    fn foo(&self) {
        || self.0;
    }
}
        "#,
            expect!["56..65: self"],
        );
    }
}
