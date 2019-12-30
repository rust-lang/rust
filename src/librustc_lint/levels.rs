use rustc::hir;
use rustc::hir::def_id::{CrateNum, LOCAL_CRATE};
use rustc::hir::intravisit;
use rustc::lint::{LintLevelMap, LintLevelSets, LintLevelsBuilder, LintStore};
use rustc::ty::query::Providers;
use rustc::ty::TyCtxt;
use syntax::ast;

pub use rustc_session::lint::{FutureIncompatibleInfo, Level, Lint, LintId};

fn lint_levels(tcx: TyCtxt<'_>, cnum: CrateNum) -> &LintLevelMap {
    assert_eq!(cnum, LOCAL_CRATE);
    let store = &tcx.lint_store;
    let mut builder = LintLevelMapBuilder {
        levels: LintLevelSets::builder(tcx.sess, false, &store),
        tcx: tcx,
        store: store,
    };
    let krate = tcx.hir().krate();

    let push = builder.levels.push(&krate.attrs, &store);
    builder.levels.register_id(hir::CRATE_HIR_ID);
    for macro_def in krate.exported_macros {
        builder.levels.register_id(macro_def.hir_id);
    }
    intravisit::walk_crate(&mut builder, krate);
    builder.levels.pop(push);

    tcx.arena.alloc(builder.levels.build_map())
}

struct LintLevelMapBuilder<'a, 'tcx> {
    levels: LintLevelsBuilder<'tcx>,
    tcx: TyCtxt<'tcx>,
    store: &'a LintStore,
}

impl LintLevelMapBuilder<'_, '_> {
    fn with_lint_attrs<F>(&mut self, id: hir::HirId, attrs: &[ast::Attribute], f: F)
    where
        F: FnOnce(&mut Self),
    {
        let push = self.levels.push(attrs, self.store);
        if push.changed {
            self.levels.register_id(id);
        }
        f(self);
        self.levels.pop(push);
    }
}

impl<'tcx> intravisit::Visitor<'tcx> for LintLevelMapBuilder<'_, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> intravisit::NestedVisitorMap<'this, 'tcx> {
        intravisit::NestedVisitorMap::All(&self.tcx.hir())
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        self.with_lint_attrs(param.hir_id, &param.attrs, |builder| {
            intravisit::walk_param(builder, param);
        });
    }

    fn visit_item(&mut self, it: &'tcx hir::Item<'tcx>) {
        self.with_lint_attrs(it.hir_id, &it.attrs, |builder| {
            intravisit::walk_item(builder, it);
        });
    }

    fn visit_foreign_item(&mut self, it: &'tcx hir::ForeignItem<'tcx>) {
        self.with_lint_attrs(it.hir_id, &it.attrs, |builder| {
            intravisit::walk_foreign_item(builder, it);
        })
    }

    fn visit_expr(&mut self, e: &'tcx hir::Expr<'tcx>) {
        self.with_lint_attrs(e.hir_id, &e.attrs, |builder| {
            intravisit::walk_expr(builder, e);
        })
    }

    fn visit_struct_field(&mut self, s: &'tcx hir::StructField<'tcx>) {
        self.with_lint_attrs(s.hir_id, &s.attrs, |builder| {
            intravisit::walk_struct_field(builder, s);
        })
    }

    fn visit_variant(
        &mut self,
        v: &'tcx hir::Variant<'tcx>,
        g: &'tcx hir::Generics<'tcx>,
        item_id: hir::HirId,
    ) {
        self.with_lint_attrs(v.id, &v.attrs, |builder| {
            intravisit::walk_variant(builder, v, g, item_id);
        })
    }

    fn visit_local(&mut self, l: &'tcx hir::Local<'tcx>) {
        self.with_lint_attrs(l.hir_id, &l.attrs, |builder| {
            intravisit::walk_local(builder, l);
        })
    }

    fn visit_arm(&mut self, a: &'tcx hir::Arm<'tcx>) {
        self.with_lint_attrs(a.hir_id, &a.attrs, |builder| {
            intravisit::walk_arm(builder, a);
        })
    }

    fn visit_trait_item(&mut self, trait_item: &'tcx hir::TraitItem<'tcx>) {
        self.with_lint_attrs(trait_item.hir_id, &trait_item.attrs, |builder| {
            intravisit::walk_trait_item(builder, trait_item);
        });
    }

    fn visit_impl_item(&mut self, impl_item: &'tcx hir::ImplItem<'tcx>) {
        self.with_lint_attrs(impl_item.hir_id, &impl_item.attrs, |builder| {
            intravisit::walk_impl_item(builder, impl_item);
        });
    }
}

pub fn provide(providers: &mut Providers<'_>) {
    providers.lint_levels = lint_levels;
}
