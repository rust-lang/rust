use clippy_utils::diagnostics::span_lint_and_help;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{GenericArg, Item, ItemKind, QPath, Ty, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::GenericParamDefKind;
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for type parameters which are positioned inconsistently between
    /// a type definition and impl block. Specifically, a parameter in an impl
    /// block which has the same name as a parameter in the type def, but is in
    /// a different place.
    ///
    /// ### Why is this bad?
    /// Type parameters are determined by their position rather than name.
    /// Naming type parameters inconsistently may cause you to refer to the
    /// wrong type parameter.
    ///
    /// ### Limitations
    /// This lint only applies to impl blocks with simple generic params, e.g.
    /// `A`. If there is anything more complicated, such as a tuple, it will be
    /// ignored.
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo<A, B> {
    ///     x: A,
    ///     y: B,
    /// }
    /// // inside the impl, B refers to Foo::A
    /// impl<B, A> Foo<B, A> {}
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct Foo<A, B> {
    ///     x: A,
    ///     y: B,
    /// }
    /// impl<A, B> Foo<A, B> {}
    /// ```
    #[clippy::version = "1.63.0"]
    pub MISMATCHING_TYPE_PARAM_ORDER,
    pedantic,
    "type parameter positioned inconsistently between type def and impl block"
}
declare_lint_pass!(TypeParamMismatch => [MISMATCHING_TYPE_PARAM_ORDER]);

impl<'tcx> LateLintPass<'tcx> for TypeParamMismatch {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Impl(imp) = &item.kind
            && let TyKind::Path(QPath::Resolved(_, path)) = &imp.self_ty.kind
            && let [segment, ..] = path.segments
            && let Some(generic_args) = segment.args
            && !generic_args.args.is_empty()
            && !item.span.from_expansion()
        {
            // get the name and span of the generic parameters in the Impl
            let mut impl_params = Vec::new();
            for p in generic_args.args {
                match p {
                    GenericArg::Type(Ty {
                        kind: TyKind::Path(QPath::Resolved(_, path)),
                        ..
                    }) => impl_params.push((path.segments[0].ident.to_string(), path.span)),
                    GenericArg::Type(_) => return,
                    _ => (),
                };
            }

            // find the type that the Impl is for
            // only lint on struct/enum/union for now
            let Res::Def(DefKind::Struct | DefKind::Enum | DefKind::Union, defid) = path.res else {
                return;
            };

            // get the names of the generic parameters in the type
            let type_params = &cx.tcx.generics_of(defid).own_params;
            let type_param_names: Vec<_> = type_params
                .iter()
                .filter_map(|p| match p.kind {
                    GenericParamDefKind::Type { .. } => Some(p.name.to_string()),
                    _ => None,
                })
                .collect();
            // hashmap of name -> index for mismatch_param_name
            let type_param_names_hashmap: FxHashMap<&String, usize> = type_param_names
                .iter()
                .enumerate()
                .map(|(i, param)| (param, i))
                .collect();

            let type_name = segment.ident;
            for (i, (impl_param_name, impl_param_span)) in impl_params.iter().enumerate() {
                if mismatch_param_name(i, impl_param_name, &type_param_names_hashmap) {
                    let msg = format!(
                        "`{type_name}` has a similarly named generic type parameter `{impl_param_name}` in its declaration, but in a different order"
                    );
                    let help = format!(
                        "try `{}`, or a name that does not conflict with `{type_name}`'s generic params",
                        type_param_names[i]
                    );
                    span_lint_and_help(cx, MISMATCHING_TYPE_PARAM_ORDER, *impl_param_span, msg, None, help);
                }
            }
        }
    }
}

// Checks if impl_param_name is the same as one of type_param_names,
// and is in a different position
fn mismatch_param_name(i: usize, impl_param_name: &String, type_param_names: &FxHashMap<&String, usize>) -> bool {
    if let Some(j) = type_param_names.get(impl_param_name) {
        if i != *j {
            return true;
        }
    }
    false
}
