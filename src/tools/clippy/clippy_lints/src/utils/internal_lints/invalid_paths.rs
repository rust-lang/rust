use clippy_utils::consts::{constant_simple, Constant};
use clippy_utils::def_path_res;
use clippy_utils::diagnostics::span_lint;
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::Item;
use rustc_hir_analysis::hir_ty_to_ty;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, fast_reject::SimplifiedTypeGen, FloatTy};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Symbol;

declare_clippy_lint! {
    /// ### What it does
    /// Checks the paths module for invalid paths.
    ///
    /// ### Why is this bad?
    /// It indicates a bug in the code.
    ///
    /// ### Example
    /// None.
    pub INVALID_PATHS,
    internal,
    "invalid path"
}

declare_lint_pass!(InvalidPaths => [INVALID_PATHS]);

impl<'tcx> LateLintPass<'tcx> for InvalidPaths {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        let local_def_id = &cx.tcx.parent_module(item.hir_id());
        let mod_name = &cx.tcx.item_name(local_def_id.to_def_id());
        if_chain! {
            if mod_name.as_str() == "paths";
            if let hir::ItemKind::Const(ty, body_id) = item.kind;
            let ty = hir_ty_to_ty(cx.tcx, ty);
            if let ty::Array(el_ty, _) = &ty.kind();
            if let ty::Ref(_, el_ty, _) = &el_ty.kind();
            if el_ty.is_str();
            let body = cx.tcx.hir().body(body_id);
            let typeck_results = cx.tcx.typeck_body(body_id);
            if let Some(Constant::Vec(path)) = constant_simple(cx, typeck_results, body.value);
            let path: Vec<&str> = path
                .iter()
                .map(|x| {
                    if let Constant::Str(s) = x {
                        s.as_str()
                    } else {
                        // We checked the type of the constant above
                        unreachable!()
                    }
                })
                .collect();
            if !check_path(cx, &path[..]);
            then {
                span_lint(cx, INVALID_PATHS, item.span, "invalid path");
            }
        }
    }
}

// This is not a complete resolver for paths. It works on all the paths currently used in the paths
// module.  That's all it does and all it needs to do.
pub fn check_path(cx: &LateContext<'_>, path: &[&str]) -> bool {
    if !def_path_res(cx, path).is_empty() {
        return true;
    }

    // Some implementations can't be found by `path_to_res`, particularly inherent
    // implementations of native types. Check lang items.
    let path_syms: Vec<_> = path.iter().map(|p| Symbol::intern(p)).collect();
    let lang_items = cx.tcx.lang_items();
    // This list isn't complete, but good enough for our current list of paths.
    let incoherent_impls = [
        SimplifiedTypeGen::FloatSimplifiedType(FloatTy::F32),
        SimplifiedTypeGen::FloatSimplifiedType(FloatTy::F64),
        SimplifiedTypeGen::SliceSimplifiedType,
        SimplifiedTypeGen::StrSimplifiedType,
    ]
    .iter()
    .flat_map(|&ty| cx.tcx.incoherent_impls(ty).iter().copied());
    for item_def_id in lang_items.iter().map(|(_, def_id)| def_id).chain(incoherent_impls) {
        let lang_item_path = cx.get_def_path(item_def_id);
        if path_syms.starts_with(&lang_item_path) {
            if let [item] = &path_syms[lang_item_path.len()..] {
                if matches!(
                    cx.tcx.def_kind(item_def_id),
                    DefKind::Mod | DefKind::Enum | DefKind::Trait
                ) {
                    for child in cx.tcx.module_children(item_def_id) {
                        if child.ident.name == *item {
                            return true;
                        }
                    }
                } else {
                    for child in cx.tcx.associated_item_def_ids(item_def_id) {
                        if cx.tcx.item_name(*child) == *item {
                            return true;
                        }
                    }
                }
            }
        }
    }

    false
}
