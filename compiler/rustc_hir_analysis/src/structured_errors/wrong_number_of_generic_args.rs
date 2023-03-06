use crate::structured_errors::StructuredDiagnostic;
use rustc_errors::{
    pluralize, Applicability, Diagnostic, DiagnosticBuilder, DiagnosticId, ErrorGuaranteed,
    MultiSpan,
};
use rustc_hir as hir;
use rustc_middle::ty::{self as ty, AssocItems, AssocKind, TyCtxt};
use rustc_session::Session;
use rustc_span::def_id::DefId;
use std::iter;

use GenericArgsInfo::*;

/// Handles the `wrong number of type / lifetime / ... arguments` family of error messages.
pub struct WrongNumberOfGenericArgs<'a, 'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,

    pub(crate) angle_brackets: AngleBrackets,

    pub(crate) gen_args_info: GenericArgsInfo,

    /// Offending path segment
    pub(crate) path_segment: &'a hir::PathSegment<'a>,

    /// Generic parameters as expected by type or trait
    pub(crate) gen_params: &'a ty::Generics,

    /// Index offset into parameters. Depends on whether `Self` is included and on
    /// number of lifetime parameters in case we're processing missing or redundant
    /// type or constant arguments.
    pub(crate) params_offset: usize,

    /// Generic arguments as provided by user
    pub(crate) gen_args: &'a hir::GenericArgs<'a>,

    /// DefId of the generic type
    pub(crate) def_id: DefId,
}

// Provides information about the kind of arguments that were provided for
// the PathSegment, for which missing generic arguments were detected
#[derive(Debug)]
pub(crate) enum AngleBrackets {
    // No angle brackets were provided, but generic arguments exist in elided form
    Implied,

    // No angle brackets were provided
    Missing,

    // Angle brackets are available, but missing some generic arguments
    Available,
}

// Information about the kind of arguments that are either missing or are unexpected
#[derive(Debug)]
pub enum GenericArgsInfo {
    MissingLifetimes {
        num_missing_args: usize,
    },
    ExcessLifetimes {
        num_redundant_args: usize,
    },
    MissingTypesOrConsts {
        num_missing_args: usize,

        // type or const generic arguments can have default values
        num_default_params: usize,

        // lifetime arguments precede type and const parameters, this
        // field gives the number of generic lifetime arguments to let
        // us infer the position of type and const generic arguments
        // in the angle brackets
        args_offset: usize,
    },

    ExcessTypesOrConsts {
        num_redundant_args: usize,

        // type or const generic arguments can have default values
        num_default_params: usize,

        // lifetime arguments precede type and const parameters, this
        // field gives the number of generic lifetime arguments to let
        // us infer the position of type and const generic arguments
        // in the angle brackets
        args_offset: usize,

        // if synthetic type arguments (e.g. `impl Trait`) are specified
        synth_provided: bool,
    },
}

impl<'a, 'tcx> WrongNumberOfGenericArgs<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        gen_args_info: GenericArgsInfo,
        path_segment: &'a hir::PathSegment<'_>,
        gen_params: &'a ty::Generics,
        params_offset: usize,
        gen_args: &'a hir::GenericArgs<'a>,
        def_id: DefId,
    ) -> Self {
        let angle_brackets = if gen_args.span_ext().is_none() {
            if gen_args.is_empty() { AngleBrackets::Missing } else { AngleBrackets::Implied }
        } else {
            AngleBrackets::Available
        };

        Self {
            tcx,
            angle_brackets,
            gen_args_info,
            path_segment,
            gen_params,
            params_offset,
            gen_args,
            def_id,
        }
    }

    fn missing_lifetimes(&self) -> bool {
        match self.gen_args_info {
            MissingLifetimes { .. } | ExcessLifetimes { .. } => true,
            MissingTypesOrConsts { .. } | ExcessTypesOrConsts { .. } => false,
        }
    }

    fn kind(&self) -> &str {
        if self.missing_lifetimes() { "lifetime" } else { "generic" }
    }

    fn num_provided_args(&self) -> usize {
        if self.missing_lifetimes() {
            self.num_provided_lifetime_args()
        } else {
            self.num_provided_type_or_const_args()
        }
    }

    fn num_provided_lifetime_args(&self) -> usize {
        match self.angle_brackets {
            AngleBrackets::Missing => 0,
            // Only lifetime arguments can be implied
            AngleBrackets::Implied => self.gen_args.args.len(),
            AngleBrackets::Available => self.gen_args.num_lifetime_params(),
        }
    }

    fn num_provided_type_or_const_args(&self) -> usize {
        match self.angle_brackets {
            AngleBrackets::Missing => 0,
            // Only lifetime arguments can be implied
            AngleBrackets::Implied => 0,
            AngleBrackets::Available => self.gen_args.num_generic_params(),
        }
    }

    fn num_expected_lifetime_args(&self) -> usize {
        let num_provided_args = self.num_provided_lifetime_args();
        match self.gen_args_info {
            MissingLifetimes { num_missing_args } => num_provided_args + num_missing_args,
            ExcessLifetimes { num_redundant_args } => num_provided_args - num_redundant_args,
            _ => 0,
        }
    }

    fn num_expected_type_or_const_args(&self) -> usize {
        let num_provided_args = self.num_provided_type_or_const_args();
        match self.gen_args_info {
            MissingTypesOrConsts { num_missing_args, .. } => num_provided_args + num_missing_args,
            ExcessTypesOrConsts { num_redundant_args, .. } => {
                num_provided_args - num_redundant_args
            }
            _ => 0,
        }
    }

    // Gives the number of expected arguments taking into account default arguments
    fn num_expected_type_or_const_args_including_defaults(&self) -> usize {
        let provided_args = self.num_provided_type_or_const_args();
        match self.gen_args_info {
            MissingTypesOrConsts { num_missing_args, num_default_params, .. } => {
                provided_args + num_missing_args - num_default_params
            }
            ExcessTypesOrConsts { num_redundant_args, num_default_params, .. } => {
                provided_args - num_redundant_args - num_default_params
            }
            _ => 0,
        }
    }

    fn num_missing_lifetime_args(&self) -> usize {
        let missing_args = self.num_expected_lifetime_args() - self.num_provided_lifetime_args();
        assert!(missing_args > 0);
        missing_args
    }

    fn num_missing_type_or_const_args(&self) -> usize {
        let missing_args = self.num_expected_type_or_const_args_including_defaults()
            - self.num_provided_type_or_const_args();
        assert!(missing_args > 0);
        missing_args
    }

    fn num_excess_lifetime_args(&self) -> usize {
        match self.gen_args_info {
            ExcessLifetimes { num_redundant_args } => num_redundant_args,
            _ => 0,
        }
    }

    fn num_excess_type_or_const_args(&self) -> usize {
        match self.gen_args_info {
            ExcessTypesOrConsts { num_redundant_args, .. } => num_redundant_args,
            _ => 0,
        }
    }

    fn too_many_args_provided(&self) -> bool {
        match self.gen_args_info {
            MissingLifetimes { .. } | MissingTypesOrConsts { .. } => false,
            ExcessLifetimes { num_redundant_args }
            | ExcessTypesOrConsts { num_redundant_args, .. } => {
                assert!(num_redundant_args > 0);
                true
            }
        }
    }

    fn not_enough_args_provided(&self) -> bool {
        match self.gen_args_info {
            MissingLifetimes { num_missing_args }
            | MissingTypesOrConsts { num_missing_args, .. } => {
                assert!(num_missing_args > 0);
                true
            }
            ExcessLifetimes { .. } | ExcessTypesOrConsts { .. } => false,
        }
    }

    // Helper method to get the index offset in angle brackets, at which type or const arguments
    // start appearing
    fn get_lifetime_args_offset(&self) -> usize {
        match self.gen_args_info {
            MissingLifetimes { .. } | ExcessLifetimes { .. } => 0,
            MissingTypesOrConsts { args_offset, .. } | ExcessTypesOrConsts { args_offset, .. } => {
                args_offset
            }
        }
    }

    fn get_num_default_params(&self) -> usize {
        match self.gen_args_info {
            MissingTypesOrConsts { num_default_params, .. }
            | ExcessTypesOrConsts { num_default_params, .. } => num_default_params,
            _ => 0,
        }
    }

    fn is_synth_provided(&self) -> bool {
        match self.gen_args_info {
            ExcessTypesOrConsts { synth_provided, .. } => synth_provided,
            _ => false,
        }
    }

    // Helper function to choose a quantifier word for the number of expected arguments
    // and to give a bound for the number of expected arguments
    fn get_quantifier_and_bound(&self) -> (&'static str, usize) {
        if self.get_num_default_params() == 0 {
            match self.gen_args_info {
                MissingLifetimes { .. } | ExcessLifetimes { .. } => {
                    ("", self.num_expected_lifetime_args())
                }
                MissingTypesOrConsts { .. } | ExcessTypesOrConsts { .. } => {
                    ("", self.num_expected_type_or_const_args())
                }
            }
        } else {
            match self.gen_args_info {
                MissingLifetimes { .. } => ("at least ", self.num_expected_lifetime_args()),
                MissingTypesOrConsts { .. } => {
                    ("at least ", self.num_expected_type_or_const_args_including_defaults())
                }
                ExcessLifetimes { .. } => ("at most ", self.num_expected_lifetime_args()),
                ExcessTypesOrConsts { .. } => ("at most ", self.num_expected_type_or_const_args()),
            }
        }
    }

    // Creates lifetime name suggestions from the lifetime parameter names
    fn get_lifetime_args_suggestions_from_param_names(
        &self,
        path_hir_id: hir::HirId,
        num_params_to_take: usize,
    ) -> String {
        debug!(?path_hir_id);

        // If there was already a lifetime among the arguments, just replicate that one.
        if let Some(lt) = self.gen_args.args.iter().find_map(|arg| match arg {
            hir::GenericArg::Lifetime(lt) => Some(lt),
            _ => None,
        }) {
            return std::iter::repeat(lt.to_string())
                .take(num_params_to_take)
                .collect::<Vec<_>>()
                .join(", ");
        }

        let mut ret = Vec::new();
        let mut ty_id = None;
        for (id, node) in self.tcx.hir().parent_iter(path_hir_id) {
            debug!(?id);
            if let hir::Node::Ty(_) = node {
                ty_id = Some(id);
            }

            // Suggest `'_` when in function parameter or elided function return.
            if let Some(fn_decl) = node.fn_decl() && let Some(ty_id) = ty_id {
                let in_arg = fn_decl.inputs.iter().any(|t| t.hir_id == ty_id);
                let in_ret = matches!(fn_decl.output, hir::FnRetTy::Return(ty) if ty.hir_id == ty_id);

                if in_arg || (in_ret && fn_decl.lifetime_elision_allowed) {
                    return std::iter::repeat("'_".to_owned()).take(num_params_to_take).collect::<Vec<_>>().join(", ");
                }
            }

            // Suggest `'static` when in const/static item-like.
            if let hir::Node::Item(hir::Item {
                kind: hir::ItemKind::Static { .. } | hir::ItemKind::Const { .. },
                ..
            })
            | hir::Node::TraitItem(hir::TraitItem {
                kind: hir::TraitItemKind::Const { .. },
                ..
            })
            | hir::Node::ImplItem(hir::ImplItem {
                kind: hir::ImplItemKind::Const { .. },
                ..
            })
            | hir::Node::ForeignItem(hir::ForeignItem {
                kind: hir::ForeignItemKind::Static { .. },
                ..
            })
            | hir::Node::AnonConst(..) = node
            {
                return std::iter::repeat("'static".to_owned())
                    .take(num_params_to_take.saturating_sub(ret.len()))
                    .collect::<Vec<_>>()
                    .join(", ");
            }

            let params = if let Some(generics) = node.generics() {
                generics.params
            } else if let hir::Node::Ty(ty) = node
                && let hir::TyKind::BareFn(bare_fn) = ty.kind
            {
                bare_fn.generic_params
            } else {
                &[]
            };
            ret.extend(params.iter().filter_map(|p| {
                let hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit }
                    = p.kind
                else { return None };
                let hir::ParamName::Plain(name) = p.name else { return None };
                Some(name.to_string())
            }));

            if ret.len() >= num_params_to_take {
                return ret[..num_params_to_take].join(", ");
            }
            // We cannot refer to lifetimes defined in an outer function.
            if let hir::Node::Item(_) = node {
                break;
            }
        }

        // We could not gather enough lifetime parameters in the scope.
        // We use the parameter names from the target type's definition instead.
        self.gen_params
            .params
            .iter()
            .skip(self.params_offset + self.num_provided_lifetime_args())
            .take(num_params_to_take)
            .map(|param| param.name.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    // Creates type or constant name suggestions from the provided parameter names
    fn get_type_or_const_args_suggestions_from_param_names(
        &self,
        num_params_to_take: usize,
    ) -> String {
        let fn_sig = self.tcx.hir().get_if_local(self.def_id).and_then(hir::Node::fn_sig);
        let is_used_in_input = |def_id| {
            fn_sig.map_or(false, |fn_sig| {
                fn_sig.decl.inputs.iter().any(|ty| match ty.kind {
                    hir::TyKind::Path(hir::QPath::Resolved(
                        None,
                        hir::Path { res: hir::def::Res::Def(_, id), .. },
                    )) => *id == def_id,
                    _ => false,
                })
            })
        };
        self.gen_params
            .params
            .iter()
            .skip(self.params_offset + self.num_provided_type_or_const_args())
            .take(num_params_to_take)
            .map(|param| match param.kind {
                // This is being inferred from the item's inputs, no need to set it.
                ty::GenericParamDefKind::Type { .. } if is_used_in_input(param.def_id) => {
                    "_".to_string()
                }
                _ => param.name.to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn get_unbound_associated_types(&self) -> Vec<String> {
        if self.tcx.is_trait(self.def_id) {
            let items: &AssocItems = self.tcx.associated_items(self.def_id);
            items
                .in_definition_order()
                .filter(|item| item.kind == AssocKind::Type)
                .filter(|item| {
                    !self.gen_args.bindings.iter().any(|binding| binding.ident.name == item.name)
                })
                .map(|item| item.name.to_ident_string())
                .collect()
        } else {
            Vec::default()
        }
    }

    fn create_error_message(&self) -> String {
        let def_path = self.tcx.def_path_str(self.def_id);
        let def_kind = self.tcx.def_descr(self.def_id);
        let (quantifier, bound) = self.get_quantifier_and_bound();
        let kind = self.kind();
        let provided_lt_args = self.num_provided_lifetime_args();
        let provided_type_or_const_args = self.num_provided_type_or_const_args();

        let (provided_args_str, verb) = match self.gen_args_info {
            MissingLifetimes { .. } | ExcessLifetimes { .. } => (
                format!("{} lifetime argument{}", provided_lt_args, pluralize!(provided_lt_args)),
                pluralize!("was", provided_lt_args),
            ),
            MissingTypesOrConsts { .. } | ExcessTypesOrConsts { .. } => (
                format!(
                    "{} generic argument{}",
                    provided_type_or_const_args,
                    pluralize!(provided_type_or_const_args)
                ),
                pluralize!("was", provided_type_or_const_args),
            ),
        };

        if self.gen_args.span_ext().is_some() {
            format!(
                "{} takes {}{} {} argument{} but {} {} supplied",
                def_kind,
                quantifier,
                bound,
                kind,
                pluralize!(bound),
                provided_args_str.as_str(),
                verb
            )
        } else {
            format!("missing generics for {} `{}`", def_kind, def_path)
        }
    }

    fn start_diagnostics(&self) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let span = self.path_segment.ident.span;
        let msg = self.create_error_message();

        self.tcx.sess.struct_span_err_with_code(span, &msg, self.code())
    }

    /// Builds the `expected 1 type argument / supplied 2 type arguments` message.
    fn notify(&self, err: &mut Diagnostic) {
        let (quantifier, bound) = self.get_quantifier_and_bound();
        let provided_args = self.num_provided_args();

        err.span_label(
            self.path_segment.ident.span,
            format!(
                "expected {}{} {} argument{}",
                quantifier,
                bound,
                self.kind(),
                pluralize!(bound),
            ),
        );

        // When too many arguments were provided, we don't highlight each of them, because it
        // would overlap with the suggestion to remove them:
        //
        // ```
        // type Foo = Bar<usize, usize>;
        //                -----  ----- supplied 2 type arguments
        //                     ^^^^^^^ remove this type argument
        // ```
        if self.too_many_args_provided() {
            return;
        }

        let args = self
            .gen_args
            .args
            .iter()
            .skip(self.get_lifetime_args_offset())
            .take(provided_args)
            .enumerate();

        for (i, arg) in args {
            err.span_label(
                arg.span(),
                if i + 1 == provided_args {
                    format!(
                        "supplied {} {} argument{}",
                        provided_args,
                        self.kind(),
                        pluralize!(provided_args)
                    )
                } else {
                    String::new()
                },
            );
        }
    }

    fn suggest(&self, err: &mut Diagnostic) {
        debug!(
            "suggest(self.provided {:?}, self.gen_args.span(): {:?})",
            self.num_provided_args(),
            self.gen_args.span(),
        );

        match self.angle_brackets {
            AngleBrackets::Missing | AngleBrackets::Implied => self.suggest_adding_args(err),
            AngleBrackets::Available => {
                if self.not_enough_args_provided() {
                    self.suggest_adding_args(err);
                } else if self.too_many_args_provided() {
                    self.suggest_moving_args_from_assoc_fn_to_trait(err);
                    self.suggest_removing_args_or_generics(err);
                } else {
                    unreachable!();
                }
            }
        }
    }

    /// Suggests to add missing argument(s) when current invocation site already contains some
    /// generics:
    ///
    /// ```text
    /// type Map = HashMap<String>;
    /// ```
    fn suggest_adding_args(&self, err: &mut Diagnostic) {
        if self.gen_args.parenthesized {
            return;
        }

        match self.gen_args_info {
            MissingLifetimes { .. } => {
                self.suggest_adding_lifetime_args(err);
            }
            MissingTypesOrConsts { .. } => {
                self.suggest_adding_type_and_const_args(err);
            }
            _ => unreachable!(),
        }
    }

    fn suggest_adding_lifetime_args(&self, err: &mut Diagnostic) {
        debug!("suggest_adding_lifetime_args(path_segment: {:?})", self.path_segment);
        let num_missing_args = self.num_missing_lifetime_args();
        let num_params_to_take = num_missing_args;
        let msg = format!("add missing {} argument{}", self.kind(), pluralize!(num_missing_args));

        let suggested_args = self.get_lifetime_args_suggestions_from_param_names(
            self.path_segment.hir_id,
            num_params_to_take,
        );
        debug!("suggested_args: {:?}", &suggested_args);

        match self.angle_brackets {
            AngleBrackets::Missing => {
                let span = self.path_segment.ident.span;

                // insert a suggestion of the form "Y<'a, 'b>"
                let sugg = format!("<{}>", suggested_args);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    &msg,
                    sugg,
                    Applicability::HasPlaceholders,
                );
            }

            AngleBrackets::Available => {
                let (sugg_span, is_first) = if self.num_provided_lifetime_args() == 0 {
                    (self.gen_args.span().unwrap().shrink_to_lo(), true)
                } else {
                    let last_lt = &self.gen_args.args[self.num_provided_lifetime_args() - 1];
                    (last_lt.span().shrink_to_hi(), false)
                };
                let has_non_lt_args = self.num_provided_type_or_const_args() != 0;
                let has_bindings = !self.gen_args.bindings.is_empty();

                let sugg_prefix = if is_first { "" } else { ", " };
                let sugg_suffix =
                    if is_first && (has_non_lt_args || has_bindings) { ", " } else { "" };

                let sugg = format!("{}{}{}", sugg_prefix, suggested_args, sugg_suffix);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(sugg_span, &msg, sugg, Applicability::HasPlaceholders);
            }
            AngleBrackets::Implied => {
                // We never encounter missing lifetimes in situations in which lifetimes are elided
                unreachable!();
            }
        }
    }

    fn suggest_adding_type_and_const_args(&self, err: &mut Diagnostic) {
        let num_missing_args = self.num_missing_type_or_const_args();
        let msg = format!("add missing {} argument{}", self.kind(), pluralize!(num_missing_args));

        let suggested_args =
            self.get_type_or_const_args_suggestions_from_param_names(num_missing_args);
        debug!("suggested_args: {:?}", suggested_args);

        match self.angle_brackets {
            AngleBrackets::Missing | AngleBrackets::Implied => {
                let span = self.path_segment.ident.span;

                // insert a suggestion of the form "Y<T, U>"
                let sugg = format!("<{}>", suggested_args);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(
                    span.shrink_to_hi(),
                    &msg,
                    sugg,
                    Applicability::HasPlaceholders,
                );
            }
            AngleBrackets::Available => {
                let gen_args_span = self.gen_args.span().unwrap();
                let sugg_offset =
                    self.get_lifetime_args_offset() + self.num_provided_type_or_const_args();

                let (sugg_span, is_first) = if sugg_offset == 0 {
                    (gen_args_span.shrink_to_lo(), true)
                } else {
                    let arg_span = self.gen_args.args[sugg_offset - 1].span();
                    // If we came here then inferred lifetime's spans can only point
                    // to either the opening bracket or to the space right after.
                    // Both of these spans have an `hi` lower than or equal to the span
                    // of the generics excluding the brackets.
                    // This allows us to check if `arg_span` is the artificial span of
                    // an inferred lifetime, in which case the generic we're suggesting to
                    // add will be the first visible, even if it isn't the actual first generic.
                    (arg_span.shrink_to_hi(), arg_span.hi() <= gen_args_span.lo())
                };

                let sugg_prefix = if is_first { "" } else { ", " };
                let sugg_suffix =
                    if is_first && !self.gen_args.bindings.is_empty() { ", " } else { "" };

                let sugg = format!("{}{}{}", sugg_prefix, suggested_args, sugg_suffix);
                debug!("sugg: {:?}", sugg);

                err.span_suggestion_verbose(sugg_span, &msg, sugg, Applicability::HasPlaceholders);
            }
        }
    }

    /// Suggests moving redundant argument(s) of an associate function to the
    /// trait it belongs to.
    ///
    /// ```compile_fail
    /// Into::into::<Option<_>>(42) // suggests considering `Into::<Option<_>>::into(42)`
    /// ```
    fn suggest_moving_args_from_assoc_fn_to_trait(&self, err: &mut Diagnostic) {
        let trait_ = match self.tcx.trait_of_item(self.def_id) {
            Some(def_id) => def_id,
            None => return,
        };

        // Skip suggestion when the associated function is itself generic, it is unclear
        // how to split the provided parameters between those to suggest to the trait and
        // those to remain on the associated type.
        let num_assoc_fn_expected_args =
            self.num_expected_type_or_const_args() + self.num_expected_lifetime_args();
        if num_assoc_fn_expected_args > 0 {
            return;
        }

        let num_assoc_fn_excess_args =
            self.num_excess_type_or_const_args() + self.num_excess_lifetime_args();

        let trait_generics = self.tcx.generics_of(trait_);
        let num_trait_generics_except_self =
            trait_generics.count() - if trait_generics.has_self { 1 } else { 0 };

        let msg = format!(
            "consider moving {these} generic argument{s} to the `{name}` trait, which takes up to {num} argument{s}",
            these = pluralize!("this", num_assoc_fn_excess_args),
            s = pluralize!(num_assoc_fn_excess_args),
            name = self.tcx.item_name(trait_),
            num = num_trait_generics_except_self,
        );

        if let Some(parent_node) = self.tcx.hir().opt_parent_id(self.path_segment.hir_id)
        && let Some(parent_node) = self.tcx.hir().find(parent_node)
        && let hir::Node::Expr(expr) = parent_node {
            match &expr.kind {
                hir::ExprKind::Path(qpath) => {
                    self.suggest_moving_args_from_assoc_fn_to_trait_for_qualified_path(
                        err,
                        qpath,
                        msg,
                        num_assoc_fn_excess_args,
                        num_trait_generics_except_self
                    )
                },
                hir::ExprKind::MethodCall(..) => {
                    self.suggest_moving_args_from_assoc_fn_to_trait_for_method_call(
                        err,
                        trait_,
                        expr,
                        msg,
                        num_assoc_fn_excess_args,
                        num_trait_generics_except_self
                    )
                },
                _ => return,
            }
        }
    }

    fn suggest_moving_args_from_assoc_fn_to_trait_for_qualified_path(
        &self,
        err: &mut Diagnostic,
        qpath: &'tcx hir::QPath<'tcx>,
        msg: String,
        num_assoc_fn_excess_args: usize,
        num_trait_generics_except_self: usize,
    ) {
        if let hir::QPath::Resolved(_, path) = qpath
        && let Some(trait_path_segment) = path.segments.get(0) {
            let num_generic_args_supplied_to_trait = trait_path_segment.args().num_generic_params();

            if num_generic_args_supplied_to_trait + num_assoc_fn_excess_args == num_trait_generics_except_self
            {
                if let Some(span) = self.gen_args.span_ext()
                && let Ok(snippet) = self.tcx.sess.source_map().span_to_snippet(span) {
                    let sugg = vec![
                        (self.path_segment.ident.span, format!("{}::{}", snippet, self.path_segment.ident)),
                        (span.with_lo(self.path_segment.ident.span.hi()), "".to_owned())
                    ];

                    err.multipart_suggestion(
                        msg,
                        sugg,
                        Applicability::MaybeIncorrect
                    );
                }
            }
        }
    }

    fn suggest_moving_args_from_assoc_fn_to_trait_for_method_call(
        &self,
        err: &mut Diagnostic,
        trait_def_id: DefId,
        expr: &'tcx hir::Expr<'tcx>,
        msg: String,
        num_assoc_fn_excess_args: usize,
        num_trait_generics_except_self: usize,
    ) {
        let sm = self.tcx.sess.source_map();
        let hir::ExprKind::MethodCall(_, rcvr, args, _) = expr.kind else { return; };
        if num_assoc_fn_excess_args != num_trait_generics_except_self {
            return;
        }
        let Some(gen_args) = self.gen_args.span_ext() else { return; };
        let Ok(generics) = sm.span_to_snippet(gen_args) else { return; };
        let Ok(rcvr) = sm.span_to_snippet(
            rcvr.span.find_ancestor_inside(expr.span).unwrap_or(rcvr.span)
        ) else { return; };
        let Ok(rest) =
            (match args {
                [] => Ok(String::new()),
                [arg] => sm.span_to_snippet(
                    arg.span.find_ancestor_inside(expr.span).unwrap_or(arg.span),
                ),
                [first, .., last] => {
                    let first_span =
                        first.span.find_ancestor_inside(expr.span).unwrap_or(first.span);
                    let last_span =
                        last.span.find_ancestor_inside(expr.span).unwrap_or(last.span);
                    sm.span_to_snippet(first_span.to(last_span))
                }
            }) else { return; };
        let comma = if args.len() > 0 { ", " } else { "" };
        let trait_path = self.tcx.def_path_str(trait_def_id);
        let method_name = self.tcx.item_name(self.def_id);
        err.span_suggestion(
            expr.span,
            msg,
            format!("{trait_path}::{generics}::{method_name}({rcvr}{comma}{rest})"),
            Applicability::MaybeIncorrect,
        );
    }

    /// Suggests to remove redundant argument(s):
    ///
    /// ```text
    /// type Map = HashMap<String, String, String, String>;
    /// ```
    fn suggest_removing_args_or_generics(&self, err: &mut Diagnostic) {
        let num_provided_lt_args = self.num_provided_lifetime_args();
        let num_provided_type_const_args = self.num_provided_type_or_const_args();
        let unbound_types = self.get_unbound_associated_types();
        let num_provided_args = num_provided_lt_args + num_provided_type_const_args;
        assert!(num_provided_args > 0);

        let num_redundant_lt_args = self.num_excess_lifetime_args();
        let num_redundant_type_or_const_args = self.num_excess_type_or_const_args();
        let num_redundant_args = num_redundant_lt_args + num_redundant_type_or_const_args;

        let redundant_lifetime_args = num_redundant_lt_args > 0;
        let redundant_type_or_const_args = num_redundant_type_or_const_args > 0;

        let remove_entire_generics = num_redundant_args >= self.gen_args.args.len();
        let provided_args_matches_unbound_traits =
            unbound_types.len() == num_redundant_type_or_const_args;

        let remove_lifetime_args = |err: &mut Diagnostic| {
            let mut lt_arg_spans = Vec::new();
            let mut found_redundant = false;
            for arg in self.gen_args.args {
                if let hir::GenericArg::Lifetime(_) = arg {
                    lt_arg_spans.push(arg.span());
                    if lt_arg_spans.len() > self.num_expected_lifetime_args() {
                        found_redundant = true;
                    }
                } else if found_redundant {
                    // Argument which is redundant and separated like this `'c`
                    // is not included to avoid including `Bar` in span.
                    // ```
                    // type Foo<'a, T> = &'a T;
                    // let _: Foo<'a, 'b, Bar, 'c>;
                    // ```
                    break;
                }
            }

            let span_lo_redundant_lt_args = lt_arg_spans[self.num_expected_lifetime_args()];
            let span_hi_redundant_lt_args = lt_arg_spans[lt_arg_spans.len() - 1];

            let span_redundant_lt_args = span_lo_redundant_lt_args.to(span_hi_redundant_lt_args);
            debug!("span_redundant_lt_args: {:?}", span_redundant_lt_args);

            let num_redundant_lt_args = lt_arg_spans.len() - self.num_expected_lifetime_args();
            let msg_lifetimes = format!(
                "remove {these} lifetime argument{s}",
                these = pluralize!("this", num_redundant_lt_args),
                s = pluralize!(num_redundant_lt_args),
            );

            err.span_suggestion(
                span_redundant_lt_args,
                &msg_lifetimes,
                "",
                Applicability::MaybeIncorrect,
            );
        };

        let remove_type_or_const_args = |err: &mut Diagnostic| {
            let mut gen_arg_spans = Vec::new();
            let mut found_redundant = false;
            for arg in self.gen_args.args {
                match arg {
                    hir::GenericArg::Type(_)
                    | hir::GenericArg::Const(_)
                    | hir::GenericArg::Infer(_) => {
                        gen_arg_spans.push(arg.span());
                        if gen_arg_spans.len() > self.num_expected_type_or_const_args() {
                            found_redundant = true;
                        }
                    }
                    _ if found_redundant => break,
                    _ => {}
                }
            }

            let span_lo_redundant_type_or_const_args =
                gen_arg_spans[self.num_expected_type_or_const_args()];
            let span_hi_redundant_type_or_const_args = gen_arg_spans[gen_arg_spans.len() - 1];

            let span_redundant_type_or_const_args =
                span_lo_redundant_type_or_const_args.to(span_hi_redundant_type_or_const_args);
            debug!("span_redundant_type_or_const_args: {:?}", span_redundant_type_or_const_args);

            let num_redundant_gen_args =
                gen_arg_spans.len() - self.num_expected_type_or_const_args();
            let msg_types_or_consts = format!(
                "remove {these} generic argument{s}",
                these = pluralize!("this", num_redundant_gen_args),
                s = pluralize!(num_redundant_gen_args),
            );

            err.span_suggestion(
                span_redundant_type_or_const_args,
                &msg_types_or_consts,
                "",
                Applicability::MaybeIncorrect,
            );
        };

        // If there is a single unbound associated type and a single excess generic param
        // suggest replacing the generic param with the associated type bound
        if provided_args_matches_unbound_traits && !unbound_types.is_empty() {
            let unused_generics = &self.gen_args.args[self.num_expected_type_or_const_args()..];
            let suggestions = iter::zip(unused_generics, &unbound_types)
                .map(|(potential, name)| (potential.span().shrink_to_lo(), format!("{name} = ")))
                .collect::<Vec<_>>();

            if !suggestions.is_empty() {
                err.multipart_suggestion_verbose(
                    &format!(
                        "replace the generic bound{s} with the associated type{s}",
                        s = pluralize!(unbound_types.len())
                    ),
                    suggestions,
                    Applicability::MaybeIncorrect,
                );
            }
        } else if remove_entire_generics {
            let span = self
                .path_segment
                .args
                .unwrap()
                .span_ext()
                .unwrap()
                .with_lo(self.path_segment.ident.span.hi());

            let msg = format!(
                "remove these {}generics",
                if self.gen_args.parenthesized { "parenthetical " } else { "" },
            );

            err.span_suggestion(span, &msg, "", Applicability::MaybeIncorrect);
        } else if redundant_lifetime_args && redundant_type_or_const_args {
            remove_lifetime_args(err);
            remove_type_or_const_args(err);
        } else if redundant_lifetime_args {
            remove_lifetime_args(err);
        } else {
            assert!(redundant_type_or_const_args);
            remove_type_or_const_args(err);
        }
    }

    /// Builds the `type defined here` message.
    fn show_definition(&self, err: &mut Diagnostic) {
        let mut spans: MultiSpan = if let Some(def_span) = self.tcx.def_ident_span(self.def_id) {
            if self.tcx.sess.source_map().is_span_accessible(def_span) {
                def_span.into()
            } else {
                return;
            }
        } else {
            return;
        };

        let msg = {
            let def_kind = self.tcx.def_descr(self.def_id);
            let (quantifier, bound) = self.get_quantifier_and_bound();

            let params = if bound == 0 {
                String::new()
            } else {
                let params = self
                    .gen_params
                    .params
                    .iter()
                    .skip(self.params_offset)
                    .take(bound)
                    .map(|param| {
                        let span = self.tcx.def_span(param.def_id);
                        spans.push_span_label(span, "");
                        param
                    })
                    .map(|param| format!("`{}`", param.name))
                    .collect::<Vec<_>>()
                    .join(", ");

                format!(": {}", params)
            };

            format!(
                "{} defined here, with {}{} {} parameter{}{}",
                def_kind,
                quantifier,
                bound,
                self.kind(),
                pluralize!(bound),
                params,
            )
        };

        err.span_note(spans, &msg);
    }

    /// Add note if `impl Trait` is explicitly specified.
    fn note_synth_provided(&self, err: &mut Diagnostic) {
        if !self.is_synth_provided() {
            return;
        }

        err.note("`impl Trait` cannot be explicitly specified as a generic argument");
    }
}

impl<'tcx> StructuredDiagnostic<'tcx> for WrongNumberOfGenericArgs<'_, 'tcx> {
    fn session(&self) -> &Session {
        self.tcx.sess
    }

    fn code(&self) -> DiagnosticId {
        rustc_errors::error_code!(E0107)
    }

    fn diagnostic_common(&self) -> DiagnosticBuilder<'tcx, ErrorGuaranteed> {
        let mut err = self.start_diagnostics();

        self.notify(&mut err);
        self.suggest(&mut err);
        self.show_definition(&mut err);
        self.note_synth_provided(&mut err);

        err
    }
}
