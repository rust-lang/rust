use std::ops::DerefMut;
use std::panic;

use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_data_structures::sync::Lrc;
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use smallvec::{Array, SmallVec, smallvec};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::ptr::P;
use crate::token::{self, Token};
use crate::tokenstream::*;

macro_rules! macro_if {
    ($_: tt { $($if: tt)* } $(else {$($else: tt)*})?) => {
        $($if)*
    };
    ({ $($if: tt)* } $(else {$($else: tt)*})?) => {
        $($($else)*)?
    };
}

macro_rules! make_ast_visitor {
    ($trait: ident $(<$lt: lifetime>)? $(, $mut: ident)?) => {
        macro_rules! ref_t {
            ($t: ty) => { & $($lt)? $($mut)? $t };
        }

        macro_rules! result {
            ($V: ty) => {
                macro_if!($($mut)? { () } else { <$V>::Result })
            };
            () => {
                result!(Self)
            };
        }

        macro_rules! return_result {
            ($V: ty) => {
                macro_if!($($mut)? { () } else { <$V>::Result::output() })
            };
        }

        macro_rules! make_visit {
            (
                $ty: ty
                $$(, $$($arg: ident)? $$(_ $ignored_arg: ident)?: $arg_ty: ty)*;
                $visit: ident, $walk: ident
            ) => {
                fn $visit(
                    &mut self,
                    node: ref_t!($ty)
                    $$(, $$($arg)? $$(#[allow(unused)] $ignored_arg)?: $arg_ty)*
                ) -> result!() {
                    $walk(self, node $$($$(, $arg)?)*)
                }
            };
        }

        macro_rules! P {
            ($t: ty) => {
                macro_if!{$($mut)? {
                    P<$t>
                } else {
                    $t
                }}
            };
        }

        #[allow(unused)]
        macro_rules! deref_P {
            ($p: expr) => {
                macro_if!{$($mut)? {
                    $p.deref_mut()
                } else {
                    $p
                }}
            };
        }

        macro_rules! visit_id {
            ($vis: ident, $id: ident) => {
                macro_if!{ $($mut)? {
                    $vis.visit_id($id)
                } else {
                    // assign to _ to prevent unused_variable warnings
                    {let _ = (&$vis, &$id);}
                }}
            };
        }

        macro_rules! visit_span {
            ($vis: ident, $span: ident) => {
                macro_if!{ $($mut)? {
                    $vis.visit_span($span)
                } else {
                    // assign to _ to prevent unused_variable warnings
                    {let _ = (&$vis, &$span);}
                }}
            };
        }

        macro_rules! try_v {
            ($visit: expr) => {
                macro_if!{$($mut)? { $visit } else { try_visit!($visit) }}
            };
        }

        macro_rules! visit_o {
            ($opt: expr, $fn: expr) => {
                if let Some(elem) = $opt {
                    try_v!($fn(elem))
                }
            };
        }

        macro_rules! visit_list {
            ($visitor: expr, $visit: ident, $flat_map: ident, $list: expr $$(; $$($arg: expr),*)?) => {
                macro_if!{$($mut)? {
                    $list.flat_map_in_place(|x| $visitor.$flat_map(x $$(, $$($arg),*)?))
                } else {
                    visit_list!($visitor, $visit, $list $$(; $$($arg),*)?)
                }}
            };
            ($visitor: expr, $visit: ident, $list: expr $$(; $$($arg: expr),*)?) => {
                for elem in $list {
                    try_v!($visitor.$visit(elem $$(, $$($arg),*)?));
                }
            };
        }

        /// Each method of the traits `Visitor` and `MutVisitor` trait is a hook
        /// to be potentially overridden. Each method's default implementation
        /// recursively visits the substructure of the input via the corresponding
        /// `walk` method; e.g., the `visit_item` method by default calls `walk_item`.
        ///
        /// If you want to ensure that your code handles every variant explicitly,
        /// you need to override each method. (And you also need to monitor
        /// future changes to this trait in case a new method with a new default
        /// implementation gets introduced.)
        pub trait $trait$(<$lt>)?: Sized {
            macro_if!{$($mut)? {
                /// Mutable token visiting only exists for the `macro_rules` token marker and should not be
                /// used otherwise. Token visitor would be entirely separate from the regular visitor if
                /// the marker didn't have to visit AST fragments in nonterminal tokens.
                const VISIT_TOKENS: bool = false;

                // Methods in this trait have one of three forms:
                //
                //   fn visit_t(&mut self, t: &mut T);                      // common
                //   fn flat_map_t(&mut self, t: T) -> SmallVec<[T; 1]>;    // rare
                //   fn filter_map_t(&mut self, t: T) -> Option<T>;         // rarest
                //
                // Any additions to this trait should happen in form of a call to a public
                // `noop_*` function that only calls out to the visitor again, not other
                // `noop_*` functions. This is a necessary API workaround to the problem of
                // not being able to call out to the super default method in an overridden
                // default method.
                //
                // When writing these methods, it is better to use destructuring like this:
                //
                //   fn visit_abc(&mut self, ABC { a, b, c: _ }: &mut ABC) {
                //       visit_a(a);
                //       visit_b(b);
                //   }
                //
                // than to use field access like this:
                //
                //   fn visit_abc(&mut self, abc: &mut ABC) {
                //       visit_a(&mut abc.a);
                //       visit_b(&mut abc.b);
                //       // ignore abc.c
                //   }
                //
                // As well as being more concise, the former is explicit about which fields
                // are skipped. Furthermore, if a new field is added, the destructuring
                // version will cause a compile error, which is good. In comparison, the
                // field access version will continue working and it would be easy to
                // forget to add handling for it.

                make_visit!{CoroutineKind; visit_coroutine_kind, walk_coroutine_kind}
                make_visit!{FnHeader; visit_fn_header, walk_fn_header}
                make_visit!{ForeignMod; visit_foreign_mod, walk_foreign_mod}
                make_visit!{MacroDef; visit_macro_def, walk_macro_def}
                make_visit!{MetaItem; visit_meta_item, walk_meta_item}
                make_visit!{MetaItemInner; visit_meta_list_item, walk_meta_list_item}
                make_visit!{Path; visit_path, walk_path}
                make_visit!{PreciseCapturingArg; visit_precise_capturing_arg, walk_precise_capturing_arg}
                make_visit!{UseTree; visit_use_tree, walk_use_tree}

                fn flat_map_foreign_item(&mut self, ni: P<ForeignItem>) -> SmallVec<[P<ForeignItem>; 1]> {
                    walk_flat_map_item(self, ni)
                }

                fn flat_map_item(&mut self, i: P<Item>) -> SmallVec<[P<Item>; 1]> {
                    walk_flat_map_item(self, i)
                }

                fn flat_map_field_def(&mut self, fd: FieldDef) -> SmallVec<[FieldDef; 1]> {
                    walk_flat_map_field_def(self, fd)
                }

                fn flat_map_assoc_item(
                    &mut self,
                    i: P<AssocItem>,
                    _ctxt: AssocCtxt,
                ) -> SmallVec<[P<AssocItem>; 1]> {
                    walk_flat_map_item(self, i)
                }

                /// `Span` and `NodeId` are mutated at the caller site.
                fn visit_fn(&mut self, fk: FnKind<'_>, _: Span, _: NodeId) {
                    walk_fn(self, fk)
                }

                fn flat_map_stmt(&mut self, s: Stmt) -> SmallVec<[Stmt; 1]> {
                    walk_flat_map_stmt(self, s)
                }

                fn flat_map_arm(&mut self, arm: Arm) -> SmallVec<[Arm; 1]> {
                    walk_flat_map_arm(self, arm)
                }

                /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
                /// It can be removed once that feature is stabilized.
                fn visit_method_receiver_expr(&mut self, ex: &mut P<Expr>) {
                    self.visit_expr(ex)
                }

                fn filter_map_expr(&mut self, e: P<Expr>) -> Option<P<Expr>> {
                    noop_filter_map_expr(self, e)
                }

                fn flat_map_variant(&mut self, v: Variant) -> SmallVec<[Variant; 1]> {
                    walk_flat_map_variant(self, v)
                }

                fn flat_map_param(&mut self, param: Param) -> SmallVec<[Param; 1]> {
                    walk_flat_map_param(self, param)
                }

                fn flat_map_generic_param(&mut self, param: GenericParam) -> SmallVec<[GenericParam; 1]> {
                    walk_flat_map_generic_param(self, param)
                }

                fn flat_map_expr_field(&mut self, f: ExprField) -> SmallVec<[ExprField; 1]> {
                    walk_flat_map_expr_field(self, f)
                }

                fn visit_id(&mut self, _id: &mut NodeId) {
                    // Do nothing.
                }

                fn visit_span(&mut self, _sp: &mut Span) {
                    // Do nothing.
                }

                fn flat_map_pat_field(&mut self, fp: PatField) -> SmallVec<[PatField; 1]> {
                    walk_flat_map_pat_field(self, fp)
                }
            } else {
                /// The result type of the `visit_*` methods. Can be either `()`,
                /// or `ControlFlow<T>`.
                type Result: VisitorResult = ();

                make_visit!{AssocItem, ctxt: AssocCtxt; visit_assoc_item, walk_assoc_item}
                make_visit!{FieldDef; visit_field_def, walk_field_def}
                make_visit!{ForeignItem; visit_foreign_item, walk_item}
                make_visit!{GenericParam; visit_generic_param, walk_generic_param}
                make_visit!{Item; visit_item, walk_item}
                make_visit!{Path, _ id: NodeId; visit_path, walk_path}
                make_visit!{Stmt; visit_stmt, walk_stmt}
                make_visit!{UseTree, id: NodeId, _ nested: bool; visit_use_tree, walk_use_tree}

                /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
                /// It can be removed once that feature is stabilized.
                fn visit_method_receiver_expr(&mut self, ex: &'ast Expr) -> Self::Result {
                    self.visit_expr(ex)
                }
                fn visit_expr_post(&mut self, _ex: &'ast Expr) -> Self::Result {
                    Self::Result::output()
                }
                fn visit_fn(&mut self, fk: FnKind<'ast>, _: Span, _: NodeId) -> Self::Result {
                    walk_fn(self, fk)
                }
                fn visit_precise_capturing_arg(&mut self, arg: &'ast PreciseCapturingArg) {
                    walk_precise_capturing_arg(self, arg);
                }
                fn visit_mac_def(&mut self, _mac: &'ast MacroDef, _id: NodeId) -> Self::Result {
                    Self::Result::output()
                }
                fn visit_fn_header(&mut self, _header: &'ast FnHeader) -> Self::Result {
                    Self::Result::output()
                }
            }}

            make_visit!{AngleBracketedArgs; visit_angle_bracketed_parameter_data, walk_angle_bracketed_parameter_data}
            make_visit!{AnonConst; visit_anon_const, walk_anon_const}
            make_visit!{Arm; visit_arm, walk_arm}
            make_visit!{AssocItemConstraint; visit_assoc_item_constraint, walk_assoc_item_constraint}
            make_visit!{Attribute; visit_attribute, walk_attribute}
            make_visit!{CaptureBy; visit_capture_by, walk_capture_by}
            make_visit!{ClosureBinder; visit_closure_binder, walk_closure_binder}
            make_visit!{Crate; visit_crate, walk_crate}
            make_visit!{EnumDef; visit_enum_def, walk_enum_def}
            make_visit!{ExprField; visit_expr_field, walk_expr_field}
            make_visit!{FnDecl; visit_fn_decl, walk_fn_decl}
            make_visit!{FnRetTy; visit_fn_ret_ty, walk_fn_ret_ty}
            make_visit!{FormatArgs; visit_format_args, walk_format_args}
            make_visit!{GenericArg; visit_generic_arg, walk_generic_arg}
            make_visit!{GenericArgs; visit_generic_args, walk_generic_args}
            make_visit!{GenericBound, _ ctxt: BoundKind; visit_param_bound, walk_param_bound}
            make_visit!{Generics; visit_generics, walk_generics}
            make_visit!{Ident; visit_ident, walk_ident}
            make_visit!{InlineAsm; visit_inline_asm, walk_inline_asm}
            make_visit!{InlineAsmSym; visit_inline_asm_sym, walk_inline_asm_sym}
            make_visit!{Label; visit_label, walk_label}
            make_visit!{Lifetime, _ ctxt: LifetimeCtxt; visit_lifetime, walk_lifetime}
            make_visit!{MacCall; visit_mac_call, walk_mac}
            make_visit!{MutTy; visit_mt, walk_mt}
            make_visit!{Option<P<QSelf>>; visit_qself, walk_qself}
            make_visit!{Param; visit_param, walk_param}
            make_visit!{ParenthesizedArgs; visit_parenthesized_parameter_data, walk_parenthesized_parameter_data}
            make_visit!{PatField; visit_pat_field, walk_pat_field}
            make_visit!{PathSegment; visit_path_segment, walk_path_segment}
            make_visit!{PolyTraitRef; visit_poly_trait_ref, walk_poly_trait_ref}
            make_visit!{TraitRef; visit_trait_ref, walk_trait_ref}
            make_visit!{Variant; visit_variant, walk_variant}
            make_visit!{VariantData; visit_variant_data, walk_variant_data}
            make_visit!{Visibility; visit_vis, walk_vis}
            make_visit!{WhereClause; visit_where_clause, walk_where_clause}
            make_visit!{WherePredicate; visit_where_predicate, walk_where_predicate}

            make_visit!{P!(Block); visit_block, walk_block}
            make_visit!{P!(Expr); visit_expr, walk_expr}
            make_visit!{P!(Local); visit_local, walk_local}
            make_visit!{P!(Pat); visit_pat, walk_pat}
            make_visit!{P!(Ty); visit_ty, walk_ty}


            fn visit_variant_discr(&mut self, discr: ref_t!(AnonConst)) -> result!() {
                self.visit_anon_const(discr)
            }
        }

        pub fn walk_angle_bracketed_parameter_data<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            data: ref_t!(AngleBracketedArgs)
        ) -> result!(V) {
            let AngleBracketedArgs { args, span } = data;
            for arg in args {
                match arg {
                    AngleBracketedArg::Arg(a) => try_v!(vis.visit_generic_arg(a)),
                    AngleBracketedArg::Constraint(c) => try_v!(vis.visit_assoc_item_constraint(c)),
                }
            }
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_anon_const<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            anon_const: ref_t!(AnonConst)
        ) -> result!(V) {
            let AnonConst { id, value } = anon_const;
            try_v!(visit_id!(vis, id));
            try_v!(vis.visit_expr(value));
            return_result!(V)
        }

        pub fn walk_arm<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            arm: ref_t!(Arm)
        ) -> result!(V) {
            let Arm { attrs, pat, guard, body, span, id, is_placeholder: _ } = arm;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            try_v!(vis.visit_pat(pat));
            visit_o!(guard, |guard| vis.visit_expr(guard));
            visit_o!(body, |body| vis.visit_expr(body));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_capture_by<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            capture_by: ref_t!(CaptureBy)
        ) -> result!(V) {
            match capture_by {
                CaptureBy::Ref => {}
                CaptureBy::Value { move_kw } => {
                    try_v!(visit_span!(vis, move_kw))
                }
            }
            return_result!(V)
        }

        pub fn walk_closure_binder<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            binder: ref_t!(ClosureBinder)
        ) -> result!(V) {
            match binder {
                ClosureBinder::NotPresent => {}
                ClosureBinder::For { generic_params, span } => {
                    visit_list!(vis, visit_generic_param, flat_map_generic_param, generic_params);
                    try_v!(visit_span!(vis, span));
                }
            }
            return_result!(V)
        }

        pub fn walk_enum_def<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            enum_def: ref_t!(EnumDef)
        ) -> result!(V) {
            let EnumDef { variants } = enum_def;
            visit_list!(vis, visit_variant, flat_map_variant, variants);
            return_result!(V)
        }

        pub fn walk_expr_field<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            f: ref_t!(ExprField)
        ) -> result!(V) {
            let ExprField { ident, expr, span, is_shorthand: _, attrs, id, is_placeholder: _ } = f;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            try_v!(vis.visit_ident(ident));
            try_v!(vis.visit_expr(expr));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_fn_decl<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            decl: ref_t!(FnDecl)
        ) -> result!(V) {
            let FnDecl { inputs, output } = decl;
            visit_list!(vis, visit_param, flat_map_param, inputs);
            try_v!(vis.visit_fn_ret_ty(output));
            return_result!(V)
        }

        pub fn walk_fn_ret_ty<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            ret_ty: ref_t!(FnRetTy)
        ) -> result!(V) {
            match ret_ty {
                FnRetTy::Default(span) => { try_v!(visit_span!(vis, span)) }
                FnRetTy::Ty(output_ty) => { try_v!(vis.visit_ty(output_ty)) }
            }
            return_result!(V)
        }

        pub fn walk_format_args<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            fmt: ref_t!(FormatArgs)
        ) -> result!(V) {
            // FIXME: visit the template exhaustively.
            let FormatArgs { span, template: _, arguments } = fmt;
            let arg_iter = macro_if!{$($mut)? {
                    arguments.all_args_mut()
                } else {
                    arguments.all_args()
                }};
            for FormatArgument { kind, expr } in arg_iter {
                match kind {
                    FormatArgumentKind::Named(ident) | FormatArgumentKind::Captured(ident) => {
                        try_v!(vis.visit_ident(ident));
                    }
                    FormatArgumentKind::Normal => {}
                }
                try_v!(vis.visit_expr(expr));
            }
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_generic_args<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            generic_args: ref_t!(GenericArgs)
        ) -> result!(V) {
            match generic_args {
                GenericArgs::AngleBracketed(data) => {
                    try_v!(vis.visit_angle_bracketed_parameter_data(data));
                }
                GenericArgs::Parenthesized(data) => {
                    try_v!(vis.visit_parenthesized_parameter_data(data));
                }
                GenericArgs::ParenthesizedElided(span) => {
                    try_v!(visit_span!(vis, span));
                }
            }
            return_result!(V)
        }

        pub fn walk_generics<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            generics: ref_t!(Generics)
        ) -> result!(V) {
            let Generics { params, where_clause, span } = generics;
            visit_list!(vis, visit_generic_param, flat_map_generic_param, params);
            try_v!(vis.visit_where_clause(where_clause));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_ident<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            ident: ref_t!(Ident)
        ) -> result!(V) {
            let Ident { name: _, span } = ident;
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_label<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            label: ref_t!(Label)
        ) -> result!(V) {
            let Label { ident } = label;
            try_v!(vis.visit_ident(ident));
            return_result!(V)
        }

        pub fn walk_lifetime<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            lifetime: ref_t!(Lifetime)
        ) -> result!(V) {
            let Lifetime { id, ident } = lifetime;
            try_v!(visit_id!(vis, id));
            try_v!(vis.visit_ident(ident));
            return_result!(V)
        }

        pub fn walk_mt<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            mt: ref_t!(MutTy)
        ) -> result!(V) {
            let MutTy { ty, mutbl: _ } = mt;
            try_v!(vis.visit_ty(ty));
            return_result!(V)
        }

        pub fn walk_param<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            param: ref_t!(Param)
        ) -> result!(V) {
            let Param { attrs, id, pat, span, ty, is_placeholder: _ } = param;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            try_v!(vis.visit_pat(pat));
            try_v!(vis.visit_ty(ty));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_parenthesized_parameter_data<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            args: ref_t!(ParenthesizedArgs)
        ) -> result!(V) {
            let ParenthesizedArgs { inputs, output, span, inputs_span } = args;
            visit_list!(vis, visit_ty, inputs);
            try_v!(vis.visit_fn_ret_ty(output));
            try_v!(visit_span!(vis, span));
            try_v!(visit_span!(vis, inputs_span));
            return_result!(V)
        }

        pub fn walk_pat_field<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            fp: ref_t!(PatField)
        ) -> result!(V) {
            let PatField { attrs, id, ident, is_placeholder: _, is_shorthand: _, pat, span } = fp;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            try_v!(vis.visit_ident(ident));
            try_v!(vis.visit_pat(pat));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_poly_trait_ref<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            trait_ref: ref_t!(PolyTraitRef)
        ) -> result!(V) {
            let PolyTraitRef { bound_generic_params, trait_ref, span, modifiers: _ } = trait_ref;
            visit_list!(vis, visit_generic_param, flat_map_generic_param, bound_generic_params);
            try_v!(vis.visit_trait_ref(trait_ref));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_variant<$($lt,)? V: $trait$(<$lt>)?>(
            visitor: &mut V,
            variant: ref_t!(Variant)
        ) -> result!(V) {
            let Variant { ident, vis, attrs, id, data, disr_expr, span, is_placeholder: _ } = variant;
            try_v!(visit_id!(visitor, id));
            visit_list!(visitor, visit_attribute, attrs);
            try_v!(visitor.visit_vis(vis));
            try_v!(visitor.visit_ident(ident));
            try_v!(visitor.visit_variant_data(data));
            visit_o!(disr_expr, |disr_expr| visitor.visit_variant_discr(disr_expr));
            try_v!(visit_span!(visitor, span));
            return_result!(V)
        }

        pub fn walk_variant_data<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            vdata: ref_t!(VariantData)
        ) -> result!(V) {
            match vdata {
                VariantData::Struct { fields, recovered: _ } => {
                    visit_list!(vis, visit_field_def, flat_map_field_def, fields);
                }
                VariantData::Tuple(fields, id) => {
                    try_v!(visit_id!(vis, id));
                    visit_list!(vis, visit_field_def, flat_map_field_def, fields);
                }
                VariantData::Unit(id) => {
                    try_v!(visit_id!(vis, id));
                }
            }
            return_result!(V)
        }

        pub fn walk_where_clause<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            wc: ref_t!(WhereClause)
        ) -> result!(V) {
            let WhereClause { has_where_token: _, predicates, span } = wc;
            visit_list!(vis, visit_where_predicate, predicates);
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_where_predicate<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            predicate: ref_t!(WherePredicate)
        ) -> result!(V) {
            match predicate {
                WherePredicate::BoundPredicate(bp) => {
                    let WhereBoundPredicate { span, bound_generic_params, bounded_ty, bounds } = bp;
                    visit_list!(vis, visit_generic_param, flat_map_generic_param, bound_generic_params);
                    try_v!(vis.visit_ty(bounded_ty));
                    visit_list!(vis, visit_param_bound, bounds; BoundKind::Bound);
                    try_v!(visit_span!(vis, span));
                }
                WherePredicate::RegionPredicate(rp) => {
                    let WhereRegionPredicate { span, lifetime, bounds } = rp;
                    try_v!(vis.visit_lifetime(lifetime, LifetimeCtxt::Bound));
                    visit_list!(vis, visit_param_bound, bounds; BoundKind::Bound);
                    try_v!(visit_span!(vis, span));
                }
                WherePredicate::EqPredicate(ep) => {
                    let WhereEqPredicate { span, lhs_ty, rhs_ty } = ep;
                    try_v!(vis.visit_ty(lhs_ty));
                    try_v!(vis.visit_ty(rhs_ty));
                    try_v!(visit_span!(vis, span));
                }
            }
            return_result!(V)
        }
    }
}

pub mod visit {
    //! AST walker. Each overridden visit method has full control over what
    //! happens with its node, it can do its own traversal of the node's children,
    //! call `visit::walk_*` to apply the default traversal algorithm, or prevent
    //! deeper traversal by doing nothing.
    //!
    //! Note: it is an important invariant that the default visitor walks the body
    //! of a function in "execution order" (more concretely, reverse post-order
    //! with respect to the CFG implied by the AST), meaning that if AST node A may
    //! execute before AST node B, then A is visited first. The borrow checker in
    //! particular relies on this property.
    //!
    //! Note: walking an AST before macro expansion is probably a bad idea. For
    //! instance, a walker looking for item names in a module will miss all of
    //! those that are created by the expansion of a macro.

    pub use rustc_ast_ir::visit::VisitorResult;
    pub use rustc_ast_ir::{try_visit, visit_opt, walk_list, walk_visitable_list};

    use super::*;

    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum AssocCtxt {
        Trait,
        Impl,
    }

    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum FnCtxt {
        Free,
        Foreign,
        Assoc(AssocCtxt),
    }

    #[derive(Copy, Clone, Debug)]
    pub enum BoundKind {
        /// Trait bounds in generics bounds and type/trait alias.
        /// E.g., `<T: Bound>`, `type A: Bound`, or `where T: Bound`.
        Bound,

        /// Trait bounds in `impl` type.
        /// E.g., `type Foo = impl Bound1 + Bound2 + Bound3`.
        Impl,

        /// Trait bounds in trait object type.
        /// E.g., `dyn Bound1 + Bound2 + Bound3`.
        TraitObject,

        /// Super traits of a trait.
        /// E.g., `trait A: B`
        SuperTraits,
    }
    impl BoundKind {
        pub fn descr(self) -> &'static str {
            match self {
                BoundKind::Bound => "bounds",
                BoundKind::Impl => "`impl Trait`",
                BoundKind::TraitObject => "`dyn` trait object bounds",
                BoundKind::SuperTraits => "supertrait bounds",
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub enum FnKind<'a> {
        /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
        Fn(FnCtxt, &'a Ident, &'a FnSig, &'a Visibility, &'a Generics, Option<&'a Block>),

        /// E.g., `|x, y| body`.
        Closure(&'a ClosureBinder, &'a Option<CoroutineKind>, &'a FnDecl, &'a Expr),
    }

    impl<'a> FnKind<'a> {
        pub fn header(&self) -> Option<&'a FnHeader> {
            match *self {
                FnKind::Fn(_, _, sig, _, _, _) => Some(&sig.header),
                FnKind::Closure(..) => None,
            }
        }

        pub fn ident(&self) -> Option<&Ident> {
            match self {
                FnKind::Fn(_, ident, ..) => Some(ident),
                _ => None,
            }
        }

        pub fn decl(&self) -> &'a FnDecl {
            match self {
                FnKind::Fn(_, _, sig, _, _, _) => &sig.decl,
                FnKind::Closure(_, _, decl, _) => decl,
            }
        }

        pub fn ctxt(&self) -> Option<FnCtxt> {
            match self {
                FnKind::Fn(ctxt, ..) => Some(*ctxt),
                FnKind::Closure(..) => None,
            }
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub enum LifetimeCtxt {
        /// Appears in a reference type.
        Ref,
        /// Appears as a bound on a type or another lifetime.
        Bound,
        /// Appears as a generic argument.
        GenericArg,
    }

    pub trait WalkItemKind: Sized {
        fn walk<'a, V: Visitor<'a>>(
            &'a self,
            item: &'a Item<Self>,
            ctxt: AssocCtxt,
            visitor: &mut V,
        ) -> V::Result;
    }

    make_ast_visitor!(Visitor<'ast>);

    pub fn walk_crate<'a, V: Visitor<'a>>(visitor: &mut V, krate: &'a Crate) -> V::Result {
        let Crate { attrs, items, spans: _, id: _, is_placeholder: _ } = krate;
        walk_list!(visitor, visit_attribute, attrs);
        walk_list!(visitor, visit_item, items);
        V::Result::output()
    }

    pub fn walk_local<'a, V: Visitor<'a>>(visitor: &mut V, local: &'a Local) -> V::Result {
        let Local { id: _, pat, ty, kind, span: _, colon_sp: _, attrs, tokens: _ } = local;
        walk_list!(visitor, visit_attribute, attrs);
        try_visit!(visitor.visit_pat(pat));
        visit_opt!(visitor, visit_ty, ty);
        if let Some((init, els)) = kind.init_else_opt() {
            try_visit!(visitor.visit_expr(init));
            visit_opt!(visitor, visit_block, els);
        }
        V::Result::output()
    }

    pub fn walk_trait_ref<'a, V: Visitor<'a>>(
        visitor: &mut V,
        trait_ref: &'a TraitRef,
    ) -> V::Result {
        let TraitRef { path, ref_id } = trait_ref;
        visitor.visit_path(path, *ref_id)
    }

    impl WalkItemKind for ItemKind {
        fn walk<'a, V: Visitor<'a>>(
            &'a self,
            item: &'a Item<Self>,
            _ctxt: AssocCtxt,
            visitor: &mut V,
        ) -> V::Result {
            let Item { id, span, vis, ident, .. } = item;
            match self {
                ItemKind::ExternCrate(_rename) => {}
                ItemKind::Use(use_tree) => try_visit!(visitor.visit_use_tree(use_tree, *id, false)),
                ItemKind::Static(box StaticItem { ty, safety: _, mutability: _, expr }) => {
                    try_visit!(visitor.visit_ty(ty));
                    visit_opt!(visitor, visit_expr, expr);
                }
                ItemKind::Const(box ConstItem { defaultness: _, generics, ty, expr }) => {
                    try_visit!(visitor.visit_generics(generics));
                    try_visit!(visitor.visit_ty(ty));
                    visit_opt!(visitor, visit_expr, expr);
                }
                ItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
                    let kind = FnKind::Fn(FnCtxt::Free, ident, sig, vis, generics, body.as_deref());
                    try_visit!(visitor.visit_fn(kind, *span, *id));
                }
                ItemKind::Mod(_unsafety, mod_kind) => match mod_kind {
                    ModKind::Loaded(items, _inline, _inner_span) => {
                        walk_list!(visitor, visit_item, items);
                    }
                    ModKind::Unloaded => {}
                },
                ItemKind::ForeignMod(ForeignMod { safety: _, abi: _, items }) => {
                    walk_list!(visitor, visit_foreign_item, items);
                }
                ItemKind::GlobalAsm(asm) => try_visit!(visitor.visit_inline_asm(asm)),
                ItemKind::TyAlias(box TyAlias {
                    generics,
                    bounds,
                    ty,
                    defaultness: _,
                    where_clauses: _,
                }) => {
                    try_visit!(visitor.visit_generics(generics));
                    walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
                    visit_opt!(visitor, visit_ty, ty);
                }
                ItemKind::Enum(enum_definition, generics) => {
                    try_visit!(visitor.visit_generics(generics));
                    try_visit!(visitor.visit_enum_def(enum_definition));
                }
                ItemKind::Impl(box Impl {
                    defaultness: _,
                    safety: _,
                    generics,
                    constness: _,
                    polarity: _,
                    of_trait,
                    self_ty,
                    items,
                }) => {
                    try_visit!(visitor.visit_generics(generics));
                    visit_opt!(visitor, visit_trait_ref, of_trait);
                    try_visit!(visitor.visit_ty(self_ty));
                    walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Impl);
                }
                ItemKind::Struct(struct_definition, generics)
                | ItemKind::Union(struct_definition, generics) => {
                    try_visit!(visitor.visit_generics(generics));
                    try_visit!(visitor.visit_variant_data(struct_definition));
                }
                ItemKind::Trait(box Trait { safety: _, is_auto: _, generics, bounds, items }) => {
                    try_visit!(visitor.visit_generics(generics));
                    walk_list!(visitor, visit_param_bound, bounds, BoundKind::SuperTraits);
                    walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Trait);
                }
                ItemKind::TraitAlias(generics, bounds) => {
                    try_visit!(visitor.visit_generics(generics));
                    walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
                }
                ItemKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
                ItemKind::MacroDef(ts) => try_visit!(visitor.visit_mac_def(ts, *id)),
                ItemKind::Delegation(box Delegation {
                    id,
                    qself,
                    path,
                    rename,
                    body,
                    from_glob: _,
                }) => {
                    try_visit!(visitor.visit_qself(qself));
                    try_visit!(visitor.visit_path(path, *id));
                    visit_opt!(visitor, visit_ident, rename);
                    visit_opt!(visitor, visit_block, body);
                }
                ItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                    try_visit!(visitor.visit_qself(qself));
                    try_visit!(visitor.visit_path(prefix, *id));
                    if let Some(suffixes) = suffixes {
                        for (ident, rename) in suffixes {
                            visitor.visit_ident(ident);
                            if let Some(rename) = rename {
                                visitor.visit_ident(rename);
                            }
                        }
                    }
                    visit_opt!(visitor, visit_block, body);
                }
            }
            V::Result::output()
        }
    }

    pub fn walk_item<'a, V: Visitor<'a>>(
        visitor: &mut V,
        item: &'a Item<impl WalkItemKind>,
    ) -> V::Result {
        walk_assoc_item(visitor, item, AssocCtxt::Trait /*ignored*/)
    }

    pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) -> V::Result {
        let Ty { id, kind, span: _, tokens: _ } = typ;
        match kind {
            TyKind::Slice(ty) | TyKind::Paren(ty) => try_visit!(visitor.visit_ty(ty)),
            TyKind::Ptr(mt) => try_visit!(visitor.visit_mt(mt)),
            TyKind::Ref(opt_lifetime, mt) | TyKind::PinnedRef(opt_lifetime, mt) => {
                visit_opt!(visitor, visit_lifetime, opt_lifetime, LifetimeCtxt::Ref);
                try_visit!(visitor.visit_mt(mt));
            }
            TyKind::Tup(tuple_element_types) => {
                walk_list!(visitor, visit_ty, tuple_element_types);
            }
            TyKind::BareFn(function_declaration) => {
                let BareFnTy { safety: _, ext: _, generic_params, decl, decl_span: _ } =
                    &**function_declaration;
                walk_list!(visitor, visit_generic_param, generic_params);
                try_visit!(visitor.visit_fn_decl(decl));
            }
            TyKind::Path(maybe_qself, path) => {
                try_visit!(visitor.visit_qself(maybe_qself));
                try_visit!(visitor.visit_path(path, *id));
            }
            TyKind::Pat(ty, pat) => {
                try_visit!(visitor.visit_ty(ty));
                try_visit!(visitor.visit_pat(pat));
            }
            TyKind::Array(ty, length) => {
                try_visit!(visitor.visit_ty(ty));
                try_visit!(visitor.visit_anon_const(length));
            }
            TyKind::TraitObject(bounds, _syntax) => {
                walk_list!(visitor, visit_param_bound, bounds, BoundKind::TraitObject);
            }
            TyKind::ImplTrait(_id, bounds) => {
                walk_list!(visitor, visit_param_bound, bounds, BoundKind::Impl);
            }
            TyKind::Typeof(expression) => try_visit!(visitor.visit_anon_const(expression)),
            TyKind::Infer | TyKind::ImplicitSelf | TyKind::Dummy => {}
            TyKind::Err(_guar) => {}
            TyKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
            TyKind::Never | TyKind::CVarArgs => {}
        }
        V::Result::output()
    }

    fn walk_qself<'a, V: Visitor<'a>>(visitor: &mut V, qself: &'a Option<P<QSelf>>) -> V::Result {
        if let Some(qself) = qself {
            let QSelf { ty, path_span: _, position: _ } = &**qself;
            try_visit!(visitor.visit_ty(ty));
        }
        V::Result::output()
    }

    pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) -> V::Result {
        let Path { span: _, segments, tokens: _ } = path;
        walk_list!(visitor, visit_path_segment, segments);
        V::Result::output()
    }

    pub fn walk_use_tree<'a, V: Visitor<'a>>(
        visitor: &mut V,
        use_tree: &'a UseTree,
        id: NodeId,
    ) -> V::Result {
        let UseTree { prefix, kind, span: _ } = use_tree;
        try_visit!(visitor.visit_path(prefix, id));
        match kind {
            UseTreeKind::Simple(rename) => {
                // The extra IDs are handled during AST lowering.
                visit_opt!(visitor, visit_ident, rename);
            }
            UseTreeKind::Glob => {}
            UseTreeKind::Nested { ref items, span: _ } => {
                for &(ref nested_tree, nested_id) in items {
                    try_visit!(visitor.visit_use_tree(nested_tree, nested_id, true));
                }
            }
        }
        V::Result::output()
    }

    pub fn walk_path_segment<'a, V: Visitor<'a>>(
        visitor: &mut V,
        segment: &'a PathSegment,
    ) -> V::Result {
        let PathSegment { ident, id: _, args } = segment;
        try_visit!(visitor.visit_ident(ident));
        visit_opt!(visitor, visit_generic_args, args);
        V::Result::output()
    }

    pub fn walk_generic_arg<'a, V>(visitor: &mut V, generic_arg: &'a GenericArg) -> V::Result
    where
        V: Visitor<'a>,
    {
        match generic_arg {
            GenericArg::Lifetime(lt) => visitor.visit_lifetime(lt, LifetimeCtxt::GenericArg),
            GenericArg::Type(ty) => visitor.visit_ty(ty),
            GenericArg::Const(ct) => visitor.visit_anon_const(ct),
        }
    }

    pub fn walk_assoc_item_constraint<'a, V: Visitor<'a>>(
        visitor: &mut V,
        constraint: &'a AssocItemConstraint,
    ) -> V::Result {
        let AssocItemConstraint { id: _, ident, gen_args, kind, span: _ } = constraint;
        try_visit!(visitor.visit_ident(ident));
        visit_opt!(visitor, visit_generic_args, gen_args);
        match kind {
            AssocItemConstraintKind::Equality { term } => match term {
                Term::Ty(ty) => try_visit!(visitor.visit_ty(ty)),
                Term::Const(c) => try_visit!(visitor.visit_anon_const(c)),
            },
            AssocItemConstraintKind::Bound { bounds } => {
                walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            }
        }
        V::Result::output()
    }

    pub fn walk_pat<'a, V: Visitor<'a>>(visitor: &mut V, pattern: &'a Pat) -> V::Result {
        let Pat { id, kind, span: _, tokens: _ } = pattern;
        match kind {
            PatKind::TupleStruct(opt_qself, path, elems) => {
                try_visit!(visitor.visit_qself(opt_qself));
                try_visit!(visitor.visit_path(path, *id));
                walk_list!(visitor, visit_pat, elems);
            }
            PatKind::Path(opt_qself, path) => {
                try_visit!(visitor.visit_qself(opt_qself));
                try_visit!(visitor.visit_path(path, *id))
            }
            PatKind::Struct(opt_qself, path, fields, _rest) => {
                try_visit!(visitor.visit_qself(opt_qself));
                try_visit!(visitor.visit_path(path, *id));
                walk_list!(visitor, visit_pat_field, fields);
            }
            PatKind::Box(subpattern) | PatKind::Deref(subpattern) | PatKind::Paren(subpattern) => {
                try_visit!(visitor.visit_pat(subpattern));
            }
            PatKind::Ref(subpattern, _ /*mutbl*/) => {
                try_visit!(visitor.visit_pat(subpattern));
            }
            PatKind::Ident(_bmode, ident, optional_subpattern) => {
                try_visit!(visitor.visit_ident(ident));
                visit_opt!(visitor, visit_pat, optional_subpattern);
            }
            PatKind::Lit(expression) => try_visit!(visitor.visit_expr(expression)),
            PatKind::Range(lower_bound, upper_bound, _end) => {
                visit_opt!(visitor, visit_expr, lower_bound);
                visit_opt!(visitor, visit_expr, upper_bound);
            }
            PatKind::Wild | PatKind::Rest | PatKind::Never => {}
            PatKind::Err(_guar) => {}
            PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
                walk_list!(visitor, visit_pat, elems);
            }
            PatKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
        }
        V::Result::output()
    }

    impl WalkItemKind for ForeignItemKind {
        fn walk<'a, V: Visitor<'a>>(
            &'a self,
            item: &'a Item<Self>,
            _ctxt: AssocCtxt,
            visitor: &mut V,
        ) -> V::Result {
            let Item { id, span, ident, vis, .. } = item;
            match self {
                ForeignItemKind::Static(box StaticItem { ty, mutability: _, expr, safety: _ }) => {
                    try_visit!(visitor.visit_ty(ty));
                    visit_opt!(visitor, visit_expr, expr);
                }
                ForeignItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
                    let kind =
                        FnKind::Fn(FnCtxt::Foreign, ident, sig, vis, generics, body.as_deref());
                    try_visit!(visitor.visit_fn(kind, *span, *id));
                }
                ForeignItemKind::TyAlias(box TyAlias {
                    generics,
                    bounds,
                    ty,
                    defaultness: _,
                    where_clauses: _,
                }) => {
                    try_visit!(visitor.visit_generics(generics));
                    walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
                    visit_opt!(visitor, visit_ty, ty);
                }
                ForeignItemKind::MacCall(mac) => {
                    try_visit!(visitor.visit_mac_call(mac));
                }
            }
            V::Result::output()
        }
    }

    pub fn walk_param_bound<'a, V: Visitor<'a>>(
        visitor: &mut V,
        bound: &'a GenericBound,
    ) -> V::Result {
        match bound {
            GenericBound::Trait(typ) => visitor.visit_poly_trait_ref(typ),
            GenericBound::Outlives(lifetime) => {
                visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound)
            }
            GenericBound::Use(args, _span) => {
                walk_list!(visitor, visit_precise_capturing_arg, args);
                V::Result::output()
            }
        }
    }

    pub fn walk_precise_capturing_arg<'a, V: Visitor<'a>>(
        visitor: &mut V,
        arg: &'a PreciseCapturingArg,
    ) {
        match arg {
            PreciseCapturingArg::Lifetime(lt) => {
                visitor.visit_lifetime(lt, LifetimeCtxt::GenericArg);
            }
            PreciseCapturingArg::Arg(path, id) => {
                visitor.visit_path(path, *id);
            }
        }
    }

    pub fn walk_generic_param<'a, V: Visitor<'a>>(
        visitor: &mut V,
        param: &'a GenericParam,
    ) -> V::Result {
        let GenericParam { id: _, ident, attrs, bounds, is_placeholder: _, kind, colon_span: _ } =
            param;
        walk_list!(visitor, visit_attribute, attrs);
        try_visit!(visitor.visit_ident(ident));
        walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        match kind {
            GenericParamKind::Lifetime => (),
            GenericParamKind::Type { default } => visit_opt!(visitor, visit_ty, default),
            GenericParamKind::Const { ty, default, kw_span: _ } => {
                try_visit!(visitor.visit_ty(ty));
                visit_opt!(visitor, visit_anon_const, default);
            }
        }
        V::Result::output()
    }

    pub fn walk_fn<'a, V: Visitor<'a>>(visitor: &mut V, kind: FnKind<'a>) -> V::Result {
        match kind {
            FnKind::Fn(_ctxt, _ident, FnSig { header, decl, span: _ }, _vis, generics, body) => {
                // Identifier and visibility are visited as a part of the item.
                try_visit!(visitor.visit_fn_header(header));
                try_visit!(visitor.visit_generics(generics));
                try_visit!(visitor.visit_fn_decl(decl));
                visit_opt!(visitor, visit_block, body);
            }
            FnKind::Closure(binder, _coroutine_kind, decl, body) => {
                try_visit!(visitor.visit_closure_binder(binder));
                try_visit!(visitor.visit_fn_decl(decl));
                try_visit!(visitor.visit_expr(body));
            }
        }
        V::Result::output()
    }

    impl WalkItemKind for AssocItemKind {
        fn walk<'a, V: Visitor<'a>>(
            &'a self,
            item: &'a Item<Self>,
            ctxt: AssocCtxt,
            visitor: &mut V,
        ) -> V::Result {
            let Item { id, span, ident, vis, .. } = item;
            match self {
                AssocItemKind::Const(box ConstItem { defaultness: _, generics, ty, expr }) => {
                    try_visit!(visitor.visit_generics(generics));
                    try_visit!(visitor.visit_ty(ty));
                    visit_opt!(visitor, visit_expr, expr);
                }
                AssocItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
                    let kind =
                        FnKind::Fn(FnCtxt::Assoc(ctxt), ident, sig, vis, generics, body.as_deref());
                    try_visit!(visitor.visit_fn(kind, *span, *id));
                }
                AssocItemKind::Type(box TyAlias {
                    generics,
                    bounds,
                    ty,
                    defaultness: _,
                    where_clauses: _,
                }) => {
                    try_visit!(visitor.visit_generics(generics));
                    walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
                    visit_opt!(visitor, visit_ty, ty);
                }
                AssocItemKind::MacCall(mac) => {
                    try_visit!(visitor.visit_mac_call(mac));
                }
                AssocItemKind::Delegation(box Delegation {
                    id,
                    qself,
                    path,
                    rename,
                    body,
                    from_glob: _,
                }) => {
                    try_visit!(visitor.visit_qself(qself));
                    try_visit!(visitor.visit_path(path, *id));
                    visit_opt!(visitor, visit_ident, rename);
                    visit_opt!(visitor, visit_block, body);
                }
                AssocItemKind::DelegationMac(box DelegationMac {
                    qself,
                    prefix,
                    suffixes,
                    body,
                }) => {
                    try_visit!(visitor.visit_qself(qself));
                    try_visit!(visitor.visit_path(prefix, *id));
                    if let Some(suffixes) = suffixes {
                        for (ident, rename) in suffixes {
                            visitor.visit_ident(ident);
                            if let Some(rename) = rename {
                                visitor.visit_ident(rename);
                            }
                        }
                    }
                    visit_opt!(visitor, visit_block, body);
                }
            }
            V::Result::output()
        }
    }

    pub fn walk_assoc_item<'a, V: Visitor<'a>>(
        visitor: &mut V,
        item: &'a Item<impl WalkItemKind>,
        ctxt: AssocCtxt,
    ) -> V::Result {
        let Item { id: _, span: _, ident, vis, attrs, kind, tokens: _ } = item;
        walk_list!(visitor, visit_attribute, attrs);
        try_visit!(visitor.visit_vis(vis));
        try_visit!(visitor.visit_ident(ident));
        try_visit!(kind.walk(item, ctxt, visitor));
        V::Result::output()
    }

    pub fn walk_field_def<'a, V: Visitor<'a>>(visitor: &mut V, field: &'a FieldDef) -> V::Result {
        let FieldDef { attrs, id: _, span: _, vis, ident, ty, is_placeholder: _ } = field;
        walk_list!(visitor, visit_attribute, attrs);
        try_visit!(visitor.visit_vis(vis));
        visit_opt!(visitor, visit_ident, ident);
        try_visit!(visitor.visit_ty(ty));
        V::Result::output()
    }

    pub fn walk_block<'a, V: Visitor<'a>>(visitor: &mut V, block: &'a Block) -> V::Result {
        let Block { stmts, id: _, rules: _, span: _, tokens: _, could_be_bare_literal: _ } = block;
        walk_list!(visitor, visit_stmt, stmts);
        V::Result::output()
    }

    pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) -> V::Result {
        let Stmt { id: _, kind, span: _ } = statement;
        match kind {
            StmtKind::Let(local) => try_visit!(visitor.visit_local(local)),
            StmtKind::Item(item) => try_visit!(visitor.visit_item(item)),
            StmtKind::Expr(expr) | StmtKind::Semi(expr) => try_visit!(visitor.visit_expr(expr)),
            StmtKind::Empty => {}
            StmtKind::MacCall(mac) => {
                let MacCallStmt { mac, attrs, style: _, tokens: _ } = &**mac;
                walk_list!(visitor, visit_attribute, attrs);
                try_visit!(visitor.visit_mac_call(mac));
            }
        }
        V::Result::output()
    }

    pub fn walk_mac<'a, V: Visitor<'a>>(visitor: &mut V, mac: &'a MacCall) -> V::Result {
        let MacCall { path, args: _ } = mac;
        visitor.visit_path(path, DUMMY_NODE_ID)
    }

    pub fn walk_inline_asm<'a, V: Visitor<'a>>(visitor: &mut V, asm: &'a InlineAsm) -> V::Result {
        let InlineAsm {
            asm_macro: _,
            template: _,
            template_strs: _,
            operands,
            clobber_abis: _,
            options: _,
            line_spans: _,
        } = asm;
        for (op, _span) in operands {
            match op {
                InlineAsmOperand::In { expr, reg: _ }
                | InlineAsmOperand::Out { expr: Some(expr), reg: _, late: _ }
                | InlineAsmOperand::InOut { expr, reg: _, late: _ } => {
                    try_visit!(visitor.visit_expr(expr))
                }
                InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
                InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                    try_visit!(visitor.visit_expr(in_expr));
                    visit_opt!(visitor, visit_expr, out_expr);
                }
                InlineAsmOperand::Const { anon_const } => {
                    try_visit!(visitor.visit_anon_const(anon_const))
                }
                InlineAsmOperand::Sym { sym } => try_visit!(visitor.visit_inline_asm_sym(sym)),
                InlineAsmOperand::Label { block } => try_visit!(visitor.visit_block(block)),
            }
        }
        V::Result::output()
    }

    pub fn walk_inline_asm_sym<'a, V: Visitor<'a>>(
        visitor: &mut V,
        InlineAsmSym { id, qself, path }: &'a InlineAsmSym,
    ) -> V::Result {
        try_visit!(visitor.visit_qself(qself));
        visitor.visit_path(path, *id)
    }

    pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) -> V::Result {
        let Expr { id, kind, span, attrs, tokens: _ } = expression;
        walk_list!(visitor, visit_attribute, attrs);
        match kind {
            ExprKind::Array(subexpressions) => {
                walk_list!(visitor, visit_expr, subexpressions);
            }
            ExprKind::ConstBlock(anon_const) => try_visit!(visitor.visit_anon_const(anon_const)),
            ExprKind::Repeat(element, count) => {
                try_visit!(visitor.visit_expr(element));
                try_visit!(visitor.visit_anon_const(count));
            }
            ExprKind::Struct(se) => {
                let StructExpr { qself, path, fields, rest } = &**se;
                try_visit!(visitor.visit_qself(qself));
                try_visit!(visitor.visit_path(path, *id));
                walk_list!(visitor, visit_expr_field, fields);
                match rest {
                    StructRest::Base(expr) => try_visit!(visitor.visit_expr(expr)),
                    StructRest::Rest(_span) => {}
                    StructRest::None => {}
                }
            }
            ExprKind::Tup(subexpressions) => {
                walk_list!(visitor, visit_expr, subexpressions);
            }
            ExprKind::Call(callee_expression, arguments) => {
                try_visit!(visitor.visit_expr(callee_expression));
                walk_list!(visitor, visit_expr, arguments);
            }
            ExprKind::MethodCall(box MethodCall { seg, receiver, args, span: _ }) => {
                try_visit!(visitor.visit_expr(receiver));
                try_visit!(visitor.visit_path_segment(seg));
                walk_list!(visitor, visit_expr, args);
            }
            ExprKind::Binary(_op, left_expression, right_expression) => {
                try_visit!(visitor.visit_expr(left_expression));
                try_visit!(visitor.visit_expr(right_expression));
            }
            ExprKind::AddrOf(_kind, _mutbl, subexpression) => {
                try_visit!(visitor.visit_expr(subexpression));
            }
            ExprKind::Unary(_op, subexpression) => {
                try_visit!(visitor.visit_expr(subexpression));
            }
            ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) => {
                try_visit!(visitor.visit_expr(subexpression));
                try_visit!(visitor.visit_ty(typ));
            }
            ExprKind::Let(pat, expr, _span, _recovered) => {
                try_visit!(visitor.visit_pat(pat));
                try_visit!(visitor.visit_expr(expr));
            }
            ExprKind::If(head_expression, if_block, optional_else) => {
                try_visit!(visitor.visit_expr(head_expression));
                try_visit!(visitor.visit_block(if_block));
                visit_opt!(visitor, visit_expr, optional_else);
            }
            ExprKind::While(subexpression, block, opt_label) => {
                visit_opt!(visitor, visit_label, opt_label);
                try_visit!(visitor.visit_expr(subexpression));
                try_visit!(visitor.visit_block(block));
            }
            ExprKind::ForLoop { pat, iter, body, label, kind: _ } => {
                visit_opt!(visitor, visit_label, label);
                try_visit!(visitor.visit_pat(pat));
                try_visit!(visitor.visit_expr(iter));
                try_visit!(visitor.visit_block(body));
            }
            ExprKind::Loop(block, opt_label, _span) => {
                visit_opt!(visitor, visit_label, opt_label);
                try_visit!(visitor.visit_block(block));
            }
            ExprKind::Match(subexpression, arms, _kind) => {
                try_visit!(visitor.visit_expr(subexpression));
                walk_list!(visitor, visit_arm, arms);
            }
            ExprKind::Closure(box Closure {
                binder,
                capture_clause,
                coroutine_kind,
                constness: _,
                movability: _,
                fn_decl,
                body,
                fn_decl_span: _,
                fn_arg_span: _,
            }) => {
                try_visit!(visitor.visit_capture_by(capture_clause));
                try_visit!(visitor.visit_fn(
                    FnKind::Closure(binder, coroutine_kind, fn_decl, body),
                    *span,
                    *id
                ))
            }
            ExprKind::Block(block, opt_label) => {
                visit_opt!(visitor, visit_label, opt_label);
                try_visit!(visitor.visit_block(block));
            }
            ExprKind::Gen(_capt, body, _kind, _decl_span) => try_visit!(visitor.visit_block(body)),
            ExprKind::Await(expr, _span) => try_visit!(visitor.visit_expr(expr)),
            ExprKind::Assign(lhs, rhs, _span) => {
                try_visit!(visitor.visit_expr(lhs));
                try_visit!(visitor.visit_expr(rhs));
            }
            ExprKind::AssignOp(_op, left_expression, right_expression) => {
                try_visit!(visitor.visit_expr(left_expression));
                try_visit!(visitor.visit_expr(right_expression));
            }
            ExprKind::Field(subexpression, ident) => {
                try_visit!(visitor.visit_expr(subexpression));
                try_visit!(visitor.visit_ident(ident));
            }
            ExprKind::Index(main_expression, index_expression, _span) => {
                try_visit!(visitor.visit_expr(main_expression));
                try_visit!(visitor.visit_expr(index_expression));
            }
            ExprKind::Range(start, end, _limit) => {
                visit_opt!(visitor, visit_expr, start);
                visit_opt!(visitor, visit_expr, end);
            }
            ExprKind::Underscore => {}
            ExprKind::Path(maybe_qself, path) => {
                try_visit!(visitor.visit_qself(maybe_qself));
                try_visit!(visitor.visit_path(path, *id));
            }
            ExprKind::Break(opt_label, opt_expr) => {
                visit_opt!(visitor, visit_label, opt_label);
                visit_opt!(visitor, visit_expr, opt_expr);
            }
            ExprKind::Continue(opt_label) => {
                visit_opt!(visitor, visit_label, opt_label);
            }
            ExprKind::Ret(optional_expression) => {
                visit_opt!(visitor, visit_expr, optional_expression);
            }
            ExprKind::Yeet(optional_expression) => {
                visit_opt!(visitor, visit_expr, optional_expression);
            }
            ExprKind::Become(expr) => try_visit!(visitor.visit_expr(expr)),
            ExprKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
            ExprKind::Paren(subexpression) => try_visit!(visitor.visit_expr(subexpression)),
            ExprKind::InlineAsm(asm) => try_visit!(visitor.visit_inline_asm(asm)),
            ExprKind::FormatArgs(f) => try_visit!(visitor.visit_format_args(f)),
            ExprKind::OffsetOf(container, fields) => {
                try_visit!(visitor.visit_ty(container));
                walk_list!(visitor, visit_ident, fields.iter());
            }
            ExprKind::Yield(optional_expression) => {
                visit_opt!(visitor, visit_expr, optional_expression);
            }
            ExprKind::Try(subexpression) => try_visit!(visitor.visit_expr(subexpression)),
            ExprKind::TryBlock(body) => try_visit!(visitor.visit_block(body)),
            ExprKind::Lit(_token) => {}
            ExprKind::IncludedBytes(_bytes) => {}
            ExprKind::Err(_guar) => {}
            ExprKind::Dummy => {}
        }

        visitor.visit_expr_post(expression)
    }

    pub fn walk_vis<'a, V: Visitor<'a>>(visitor: &mut V, vis: &'a Visibility) -> V::Result {
        let Visibility { kind, span: _, tokens: _ } = vis;
        match kind {
            VisibilityKind::Restricted { path, id, shorthand: _ } => {
                try_visit!(visitor.visit_path(path, *id));
            }
            VisibilityKind::Public | VisibilityKind::Inherited => {}
        }
        V::Result::output()
    }

    pub fn walk_attribute<'a, V: Visitor<'a>>(visitor: &mut V, attr: &'a Attribute) -> V::Result {
        let Attribute { kind, id: _, style: _, span: _ } = attr;
        match kind {
            AttrKind::Normal(normal) => {
                let NormalAttr { item, tokens: _ } = &**normal;
                let AttrItem { unsafety: _, path, args, tokens: _ } = item;
                try_visit!(visitor.visit_path(path, DUMMY_NODE_ID));
                try_visit!(walk_attr_args(visitor, args));
            }
            AttrKind::DocComment(_kind, _sym) => {}
        }
        V::Result::output()
    }

    pub fn walk_attr_args<'a, V: Visitor<'a>>(visitor: &mut V, args: &'a AttrArgs) -> V::Result {
        match args {
            AttrArgs::Empty => {}
            AttrArgs::Delimited(_args) => {}
            AttrArgs::Eq(_eq_span, AttrArgsEq::Ast(expr)) => try_visit!(visitor.visit_expr(expr)),
            AttrArgs::Eq(_eq_span, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when walking mac args eq: {:?}", lit)
            }
        }
        V::Result::output()
    }
}

pub mod mut_visit {
    //! A `MutVisitor` represents an AST modification; it accepts an AST piece and
    //! mutates it in place. So, for instance, macro expansion is a `MutVisitor`
    //! that walks over an AST and modifies it.
    //!
    //! Note: using a `MutVisitor` (other than the `MacroExpander` `MutVisitor`) on
    //! an AST before macro expansion is probably a bad idea. For instance,
    //! a `MutVisitor` renaming item names in a module will miss all of those
    //! that are created by the expansion of a macro.

    use super::*;
    use crate::visit::{AssocCtxt, BoundKind, LifetimeCtxt};

    pub trait ExpectOne<A: Array> {
        fn expect_one(self, err: &'static str) -> A::Item;
    }

    impl<A: Array> ExpectOne<A> for SmallVec<A> {
        fn expect_one(self, err: &'static str) -> A::Item {
            assert!(self.len() == 1, "{}", err);
            self.into_iter().next().unwrap()
        }
    }

    pub trait WalkItemKind {
        fn walk(&mut self, span: Span, id: NodeId, visitor: &mut impl MutVisitor);
    }

    make_ast_visitor!(MutVisitor, mut);

    /// Use a map-style function (`FnOnce(T) -> T`) to overwrite a `&mut T`. Useful
    /// when using a `flat_map_*` or `filter_map_*` method within a `visit_`
    /// method.
    //
    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    pub fn visit_clobber<T: DummyAstNode>(t: &mut T, f: impl FnOnce(T) -> T) {
        let old_t = std::mem::replace(t, T::dummy());
        *t = f(old_t);
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    #[inline]
    fn visit_vec<T, F>(elems: &mut Vec<T>, mut visit_elem: F)
    where
        F: FnMut(&mut T),
    {
        for elem in elems {
            visit_elem(elem);
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    #[inline]
    fn visit_thin_vec<T, F>(elems: &mut ThinVec<T>, mut visit_elem: F)
    where
        F: FnMut(&mut T),
    {
        for elem in elems {
            visit_elem(elem);
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    #[inline]
    fn visit_opt<T, F>(opt: &mut Option<T>, mut visit_elem: F)
    where
        F: FnMut(&mut T),
    {
        if let Some(elem) = opt {
            visit_elem(elem);
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_attrs<T: MutVisitor>(vis: &mut T, attrs: &mut AttrVec) {
        for attr in attrs.iter_mut() {
            vis.visit_attribute(attr);
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    #[allow(unused)]
    fn visit_exprs<T: MutVisitor>(vis: &mut T, exprs: &mut Vec<P<Expr>>) {
        exprs.flat_map_in_place(|expr| vis.filter_map_expr(expr))
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_thin_exprs<T: MutVisitor>(vis: &mut T, exprs: &mut ThinVec<P<Expr>>) {
        exprs.flat_map_in_place(|expr| vis.filter_map_expr(expr))
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_bounds<T: MutVisitor>(vis: &mut T, bounds: &mut GenericBounds, ctxt: BoundKind) {
        visit_vec(bounds, |bound| vis.visit_param_bound(bound, ctxt));
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_attr_args<T: MutVisitor>(vis: &mut T, args: &mut AttrArgs) {
        match args {
            AttrArgs::Empty => {}
            AttrArgs::Delimited(args) => visit_delim_args(vis, args),
            AttrArgs::Eq(eq_span, AttrArgsEq::Ast(expr)) => {
                vis.visit_expr(expr);
                vis.visit_span(eq_span);
            }
            AttrArgs::Eq(_eq_span, AttrArgsEq::Hir(lit)) => {
                unreachable!("in literal form when visiting mac args eq: {:?}", lit)
            }
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_delim_args<T: MutVisitor>(vis: &mut T, args: &mut DelimArgs) {
        let DelimArgs { dspan, delim: _, tokens } = args;
        visit_tts(vis, tokens);
        visit_delim_span(vis, dspan);
    }

    pub fn visit_delim_span<T: MutVisitor>(vis: &mut T, DelimSpan { open, close }: &mut DelimSpan) {
        vis.visit_span(open);
        vis.visit_span(close);
    }

    pub fn walk_flat_map_pat_field<T: MutVisitor>(
        vis: &mut T,
        mut fp: PatField,
    ) -> SmallVec<[PatField; 1]> {
        vis.visit_pat_field(&mut fp);
        smallvec![fp]
    }

    fn walk_use_tree<T: MutVisitor>(vis: &mut T, use_tree: &mut UseTree) {
        let UseTree { prefix, kind, span } = use_tree;
        vis.visit_path(prefix);
        match kind {
            UseTreeKind::Simple(rename) => visit_opt(rename, |rename| vis.visit_ident(rename)),
            UseTreeKind::Nested { items, span } => {
                for (tree, id) in items {
                    vis.visit_id(id);
                    vis.visit_use_tree(tree);
                }
                vis.visit_span(span);
            }
            UseTreeKind::Glob => {}
        }
        vis.visit_span(span);
    }

    pub fn walk_flat_map_arm<T: MutVisitor>(vis: &mut T, mut arm: Arm) -> SmallVec<[Arm; 1]> {
        vis.visit_arm(&mut arm);
        smallvec![arm]
    }

    fn walk_assoc_item_constraint<T: MutVisitor>(
        vis: &mut T,
        AssocItemConstraint { id, ident, gen_args, kind, span }: &mut AssocItemConstraint,
    ) {
        vis.visit_id(id);
        vis.visit_ident(ident);
        if let Some(gen_args) = gen_args {
            vis.visit_generic_args(gen_args);
        }
        match kind {
            AssocItemConstraintKind::Equality { term } => match term {
                Term::Ty(ty) => vis.visit_ty(ty),
                Term::Const(c) => vis.visit_anon_const(c),
            },
            AssocItemConstraintKind::Bound { bounds } => {
                visit_bounds(vis, bounds, BoundKind::Bound)
            }
        }
        vis.visit_span(span);
    }

    pub fn walk_ty<T: MutVisitor>(vis: &mut T, ty: &mut P<Ty>) {
        let Ty { id, kind, span, tokens } = ty.deref_mut();
        vis.visit_id(id);
        match kind {
            TyKind::Err(_guar) => {}
            TyKind::Infer
            | TyKind::ImplicitSelf
            | TyKind::Dummy
            | TyKind::Never
            | TyKind::CVarArgs => {}
            TyKind::Slice(ty) => vis.visit_ty(ty),
            TyKind::Ptr(mt) => vis.visit_mt(mt),
            TyKind::Ref(lt, mt) | TyKind::PinnedRef(lt, mt) => {
                visit_opt(lt, |lt| vis.visit_lifetime(lt, LifetimeCtxt::Ref));
                vis.visit_mt(mt);
            }
            TyKind::BareFn(bft) => {
                let BareFnTy { safety, ext: _, generic_params, decl, decl_span } = bft.deref_mut();
                visit_safety(vis, safety);
                generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
                vis.visit_fn_decl(decl);
                vis.visit_span(decl_span);
            }
            TyKind::Tup(tys) => visit_thin_vec(tys, |ty| vis.visit_ty(ty)),
            TyKind::Paren(ty) => vis.visit_ty(ty),
            TyKind::Pat(ty, pat) => {
                vis.visit_ty(ty);
                vis.visit_pat(pat);
            }
            TyKind::Path(qself, path) => {
                vis.visit_qself(qself);
                vis.visit_path(path);
            }
            TyKind::Array(ty, length) => {
                vis.visit_ty(ty);
                vis.visit_anon_const(length);
            }
            TyKind::Typeof(expr) => vis.visit_anon_const(expr),
            TyKind::TraitObject(bounds, _syntax) => {
                visit_vec(bounds, |bound| vis.visit_param_bound(bound, BoundKind::TraitObject))
            }
            TyKind::ImplTrait(id, bounds) => {
                vis.visit_id(id);
                visit_vec(bounds, |bound| vis.visit_param_bound(bound, BoundKind::Impl));
            }
            TyKind::MacCall(mac) => vis.visit_mac_call(mac),
        }
        visit_lazy_tts(vis, tokens);
        vis.visit_span(span);
    }

    fn walk_foreign_mod<T: MutVisitor>(vis: &mut T, foreign_mod: &mut ForeignMod) {
        let ForeignMod { safety, abi: _, items } = foreign_mod;
        visit_safety(vis, safety);
        items.flat_map_in_place(|item| vis.flat_map_foreign_item(item));
    }

    pub fn walk_flat_map_variant<T: MutVisitor>(
        visitor: &mut T,
        mut variant: Variant,
    ) -> SmallVec<[Variant; 1]> {
        visitor.visit_variant(&mut variant);
        smallvec![variant]
    }

    fn walk_path_segment<T: MutVisitor>(vis: &mut T, segment: &mut PathSegment) {
        let PathSegment { ident, id, args } = segment;
        vis.visit_id(id);
        vis.visit_ident(ident);
        visit_opt(args, |args| vis.visit_generic_args(args));
    }

    fn walk_path<T: MutVisitor>(vis: &mut T, Path { segments, span, tokens }: &mut Path) {
        for segment in segments {
            vis.visit_path_segment(segment);
        }
        visit_lazy_tts(vis, tokens);
        vis.visit_span(span);
    }

    fn walk_qself<T: MutVisitor>(vis: &mut T, qself: &mut Option<P<QSelf>>) {
        visit_opt(qself, |qself| {
            let QSelf { ty, path_span, position: _ } = &mut **qself;
            vis.visit_ty(ty);
            vis.visit_span(path_span);
        })
    }

    fn walk_generic_arg<T: MutVisitor>(vis: &mut T, arg: &mut GenericArg) {
        match arg {
            GenericArg::Lifetime(lt) => vis.visit_lifetime(lt, LifetimeCtxt::GenericArg),
            GenericArg::Type(ty) => vis.visit_ty(ty),
            GenericArg::Const(ct) => vis.visit_anon_const(ct),
        }
    }

    fn walk_local<T: MutVisitor>(vis: &mut T, local: &mut P<Local>) {
        let Local { id, pat, ty, kind, span, colon_sp, attrs, tokens } = local.deref_mut();
        vis.visit_id(id);
        visit_attrs(vis, attrs);
        vis.visit_pat(pat);
        visit_opt(ty, |ty| vis.visit_ty(ty));
        match kind {
            LocalKind::Decl => {}
            LocalKind::Init(init) => {
                vis.visit_expr(init);
            }
            LocalKind::InitElse(init, els) => {
                vis.visit_expr(init);
                vis.visit_block(els);
            }
        }
        visit_lazy_tts(vis, tokens);
        visit_opt(colon_sp, |sp| vis.visit_span(sp));
        vis.visit_span(span);
    }

    fn walk_attribute<T: MutVisitor>(vis: &mut T, attr: &mut Attribute) {
        let Attribute { kind, id: _, style: _, span } = attr;
        match kind {
            AttrKind::Normal(normal) => {
                let NormalAttr {
                    item: AttrItem { unsafety: _, path, args, tokens },
                    tokens: attr_tokens,
                } = &mut **normal;
                vis.visit_path(path);
                visit_attr_args(vis, args);
                visit_lazy_tts(vis, tokens);
                visit_lazy_tts(vis, attr_tokens);
            }
            AttrKind::DocComment(_kind, _sym) => {}
        }
        vis.visit_span(span);
    }

    fn walk_mac<T: MutVisitor>(vis: &mut T, mac: &mut MacCall) {
        let MacCall { path, args } = mac;
        vis.visit_path(path);
        visit_delim_args(vis, args);
    }

    fn walk_macro_def<T: MutVisitor>(vis: &mut T, macro_def: &mut MacroDef) {
        let MacroDef { body, macro_rules: _ } = macro_def;
        visit_delim_args(vis, body);
    }

    fn walk_meta_list_item<T: MutVisitor>(vis: &mut T, li: &mut MetaItemInner) {
        match li {
            MetaItemInner::MetaItem(mi) => vis.visit_meta_item(mi),
            MetaItemInner::Lit(_lit) => {}
        }
    }

    fn walk_meta_item<T: MutVisitor>(vis: &mut T, mi: &mut MetaItem) {
        let MetaItem { unsafety: _, path: _, kind, span } = mi;
        match kind {
            MetaItemKind::Word => {}
            MetaItemKind::List(mis) => visit_thin_vec(mis, |mi| vis.visit_meta_list_item(mi)),
            MetaItemKind::NameValue(_s) => {}
        }
        vis.visit_span(span);
    }

    pub fn walk_flat_map_param<T: MutVisitor>(
        vis: &mut T,
        mut param: Param,
    ) -> SmallVec<[Param; 1]> {
        vis.visit_param(&mut param);
        smallvec![param]
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_attr_tt<T: MutVisitor>(vis: &mut T, tt: &mut AttrTokenTree) {
        match tt {
            AttrTokenTree::Token(token, _spacing) => {
                visit_token(vis, token);
            }
            AttrTokenTree::Delimited(dspan, _spacing, _delim, tts) => {
                visit_attr_tts(vis, tts);
                visit_delim_span(vis, dspan);
            }
            AttrTokenTree::AttrsTarget(AttrsTarget { attrs, tokens }) => {
                visit_attrs(vis, attrs);
                visit_lazy_tts_opt_mut(vis, Some(tokens));
            }
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_tt<T: MutVisitor>(vis: &mut T, tt: &mut TokenTree) {
        match tt {
            TokenTree::Token(token, _spacing) => {
                visit_token(vis, token);
            }
            TokenTree::Delimited(dspan, _spacing, _delim, tts) => {
                visit_tts(vis, tts);
                visit_delim_span(vis, dspan);
            }
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_tts<T: MutVisitor>(vis: &mut T, TokenStream(tts): &mut TokenStream) {
        if T::VISIT_TOKENS && !tts.is_empty() {
            let tts = Lrc::make_mut(tts);
            visit_vec(tts, |tree| visit_tt(vis, tree));
        }
    }

    fn visit_attr_tts<T: MutVisitor>(vis: &mut T, AttrTokenStream(tts): &mut AttrTokenStream) {
        if T::VISIT_TOKENS && !tts.is_empty() {
            let tts = Lrc::make_mut(tts);
            visit_vec(tts, |tree| visit_attr_tt(vis, tree));
        }
    }

    fn visit_lazy_tts_opt_mut<T: MutVisitor>(
        vis: &mut T,
        lazy_tts: Option<&mut LazyAttrTokenStream>,
    ) {
        if T::VISIT_TOKENS {
            if let Some(lazy_tts) = lazy_tts {
                let mut tts = lazy_tts.to_attr_token_stream();
                visit_attr_tts(vis, &mut tts);
                *lazy_tts = LazyAttrTokenStream::new(tts);
            }
        }
    }

    fn visit_lazy_tts<T: MutVisitor>(vis: &mut T, lazy_tts: &mut Option<LazyAttrTokenStream>) {
        visit_lazy_tts_opt_mut(vis, lazy_tts.as_mut());
    }

    /// Applies ident visitor if it's an ident; applies other visits to interpolated nodes.
    /// In practice the ident part is not actually used by specific visitors right now,
    /// but there's a test below checking that it works.
    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    pub fn visit_token<T: MutVisitor>(vis: &mut T, t: &mut Token) {
        let Token { kind, span } = t;
        match kind {
            token::Ident(name, _is_raw) | token::Lifetime(name, _is_raw) => {
                let mut ident = Ident::new(*name, *span);
                vis.visit_ident(&mut ident);
                *name = ident.name;
                *span = ident.span;
                return; // Avoid visiting the span for the second time.
            }
            token::NtIdent(ident, _is_raw) => {
                vis.visit_ident(ident);
            }
            token::NtLifetime(ident, _is_raw) => {
                vis.visit_ident(ident);
            }
            token::Interpolated(nt) => {
                let nt = Lrc::make_mut(nt);
                visit_nonterminal(vis, nt);
            }
            _ => {}
        }
        vis.visit_span(span);
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    /// Applies the visitor to elements of interpolated nodes.
    //
    // N.B., this can occur only when applying a visitor to partially expanded
    // code, where parsed pieces have gotten implanted ito *other* macro
    // invocations. This is relevant for macro hygiene, but possibly not elsewhere.
    //
    // One problem here occurs because the types for flat_map_item, flat_map_stmt,
    // etc., allow the visitor to return *multiple* items; this is a problem for the
    // nodes here, because they insist on having exactly one piece. One solution
    // would be to mangle the MutVisitor trait to include one-to-many and
    // one-to-one versions of these entry points, but that would probably confuse a
    // lot of people and help very few. Instead, I'm just going to put in dynamic
    // checks. I think the performance impact of this will be pretty much
    // nonexistent. The danger is that someone will apply a `MutVisitor` to a
    // partially expanded node, and will be confused by the fact that their
    // `flat_map_item` or `flat_map_stmt` isn't getting called on `NtItem` or `NtStmt`
    // nodes. Hopefully they'll wind up reading this comment, and doing something
    // appropriate.
    //
    // BTW, design choice: I considered just changing the type of, e.g., `NtItem` to
    // contain multiple items, but decided against it when I looked at
    // `parse_item_or_view_item` and tried to figure out what I would do with
    // multiple items there....
    fn visit_nonterminal<T: MutVisitor>(vis: &mut T, nt: &mut token::Nonterminal) {
        match nt {
            token::NtItem(item) => visit_clobber(item, |item| {
                // This is probably okay, because the only visitors likely to
                // peek inside interpolated nodes will be renamings/markings,
                // which map single items to single items.
                vis.flat_map_item(item).expect_one("expected visitor to produce exactly one item")
            }),
            token::NtBlock(block) => vis.visit_block(block),
            token::NtStmt(stmt) => visit_clobber(stmt, |stmt| {
                // See reasoning above.
                stmt.map(|stmt| {
                    vis.flat_map_stmt(stmt)
                        .expect_one("expected visitor to produce exactly one item")
                })
            }),
            token::NtPat(pat) => vis.visit_pat(pat),
            token::NtExpr(expr) => vis.visit_expr(expr),
            token::NtTy(ty) => vis.visit_ty(ty),
            token::NtLiteral(expr) => vis.visit_expr(expr),
            token::NtMeta(item) => {
                let AttrItem { unsafety: _, path, args, tokens } = item.deref_mut();
                vis.visit_path(path);
                visit_attr_args(vis, args);
                visit_lazy_tts(vis, tokens);
            }
            token::NtPath(path) => vis.visit_path(path),
            token::NtVis(visib) => vis.visit_vis(visib),
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_defaultness<T: MutVisitor>(vis: &mut T, defaultness: &mut Defaultness) {
        match defaultness {
            Defaultness::Default(span) => vis.visit_span(span),
            Defaultness::Final => {}
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_safety<T: MutVisitor>(vis: &mut T, safety: &mut Safety) {
        match safety {
            Safety::Unsafe(span) => vis.visit_span(span),
            Safety::Safe(span) => vis.visit_span(span),
            Safety::Default => {}
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_polarity<T: MutVisitor>(vis: &mut T, polarity: &mut ImplPolarity) {
        match polarity {
            ImplPolarity::Positive => {}
            ImplPolarity::Negative(span) => vis.visit_span(span),
        }
    }

    // No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
    fn visit_constness<T: MutVisitor>(vis: &mut T, constness: &mut Const) {
        match constness {
            Const::Yes(span) => vis.visit_span(span),
            Const::No => {}
        }
    }

    fn walk_coroutine_kind<T: MutVisitor>(vis: &mut T, coroutine_kind: &mut CoroutineKind) {
        match coroutine_kind {
            CoroutineKind::Async { span, closure_id, return_impl_trait_id }
            | CoroutineKind::Gen { span, closure_id, return_impl_trait_id }
            | CoroutineKind::AsyncGen { span, closure_id, return_impl_trait_id } => {
                vis.visit_id(closure_id);
                vis.visit_id(return_impl_trait_id);
                vis.visit_span(span);
            }
        }
    }

    fn walk_fn<T: MutVisitor>(vis: &mut T, kind: FnKind<'_>) {
        match kind {
            FnKind::Fn(FnSig { header, decl, span }, generics, body) => {
                // Identifier and visibility are visited as a part of the item.
                vis.visit_fn_header(header);
                vis.visit_generics(generics);
                vis.visit_fn_decl(decl);
                if let Some(body) = body {
                    vis.visit_block(body);
                }
                vis.visit_span(span);
            }
            FnKind::Closure(binder, decl, body) => {
                vis.visit_closure_binder(binder);
                vis.visit_fn_decl(decl);
                vis.visit_expr(body);
            }
        }
    }

    fn walk_param_bound<T: MutVisitor>(vis: &mut T, pb: &mut GenericBound) {
        match pb {
            GenericBound::Trait(ty) => vis.visit_poly_trait_ref(ty),
            GenericBound::Outlives(lifetime) => vis.visit_lifetime(lifetime, LifetimeCtxt::Bound),
            GenericBound::Use(args, span) => {
                for arg in args {
                    vis.visit_precise_capturing_arg(arg);
                }
                vis.visit_span(span);
            }
        }
    }

    fn walk_precise_capturing_arg<T: MutVisitor>(vis: &mut T, arg: &mut PreciseCapturingArg) {
        match arg {
            PreciseCapturingArg::Lifetime(lt) => {
                vis.visit_lifetime(lt, LifetimeCtxt::GenericArg);
            }
            PreciseCapturingArg::Arg(path, id) => {
                vis.visit_id(id);
                vis.visit_path(path);
            }
        }
    }

    pub fn walk_flat_map_generic_param<T: MutVisitor>(
        vis: &mut T,
        mut param: GenericParam,
    ) -> SmallVec<[GenericParam; 1]> {
        let GenericParam { id, ident, attrs, bounds, kind, colon_span, is_placeholder: _ } =
            &mut param;
        vis.visit_id(id);
        visit_attrs(vis, attrs);
        vis.visit_ident(ident);
        visit_vec(bounds, |bound| vis.visit_param_bound(bound, BoundKind::Bound));
        match kind {
            GenericParamKind::Lifetime => {}
            GenericParamKind::Type { default } => {
                visit_opt(default, |default| vis.visit_ty(default));
            }
            GenericParamKind::Const { ty, kw_span: _, default } => {
                vis.visit_ty(ty);
                visit_opt(default, |default| vis.visit_anon_const(default));
            }
        }
        if let Some(colon_span) = colon_span {
            vis.visit_span(colon_span);
        }
        smallvec![param]
    }

    fn walk_ty_alias_where_clauses<T: MutVisitor>(vis: &mut T, tawcs: &mut TyAliasWhereClauses) {
        let TyAliasWhereClauses { before, after, split: _ } = tawcs;
        let TyAliasWhereClause { has_where_token: _, span: span_before } = before;
        let TyAliasWhereClause { has_where_token: _, span: span_after } = after;
        vis.visit_span(span_before);
        vis.visit_span(span_after);
    }

    fn walk_trait_ref<T: MutVisitor>(vis: &mut T, TraitRef { path, ref_id }: &mut TraitRef) {
        vis.visit_id(ref_id);
        vis.visit_path(path);
    }

    pub fn walk_flat_map_field_def<T: MutVisitor>(
        visitor: &mut T,
        mut fd: FieldDef,
    ) -> SmallVec<[FieldDef; 1]> {
        let FieldDef { span, ident, vis, id, ty, attrs, is_placeholder: _ } = &mut fd;
        visitor.visit_id(id);
        visit_attrs(visitor, attrs);
        visitor.visit_vis(vis);
        visit_opt(ident, |ident| visitor.visit_ident(ident));
        visitor.visit_ty(ty);
        visitor.visit_span(span);
        smallvec![fd]
    }

    pub fn walk_flat_map_expr_field<T: MutVisitor>(
        vis: &mut T,
        mut f: ExprField,
    ) -> SmallVec<[ExprField; 1]> {
        vis.visit_expr_field(&mut f);
        smallvec![f]
    }

    pub fn walk_block<T: MutVisitor>(vis: &mut T, block: &mut P<Block>) {
        let Block { id, stmts, rules: _, span, tokens, could_be_bare_literal: _ } =
            block.deref_mut();
        vis.visit_id(id);
        stmts.flat_map_in_place(|stmt| vis.flat_map_stmt(stmt));
        visit_lazy_tts(vis, tokens);
        vis.visit_span(span);
    }

    pub fn walk_item_kind(
        kind: &mut impl WalkItemKind,
        span: Span,
        id: NodeId,
        vis: &mut impl MutVisitor,
    ) {
        kind.walk(span, id, vis)
    }

    impl WalkItemKind for ItemKind {
        fn walk(&mut self, span: Span, id: NodeId, vis: &mut impl MutVisitor) {
            match self {
                ItemKind::ExternCrate(_orig_name) => {}
                ItemKind::Use(use_tree) => vis.visit_use_tree(use_tree),
                ItemKind::Static(box StaticItem { ty, safety: _, mutability: _, expr }) => {
                    vis.visit_ty(ty);
                    visit_opt(expr, |expr| vis.visit_expr(expr));
                }
                ItemKind::Const(item) => {
                    visit_const_item(item, vis);
                }
                ItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
                    visit_defaultness(vis, defaultness);
                    vis.visit_fn(FnKind::Fn(sig, generics, body), span, id);
                }
                ItemKind::Mod(safety, mod_kind) => {
                    visit_safety(vis, safety);
                    match mod_kind {
                        ModKind::Loaded(
                            items,
                            _inline,
                            ModSpans { inner_span, inject_use_span },
                        ) => {
                            items.flat_map_in_place(|item| vis.flat_map_item(item));
                            vis.visit_span(inner_span);
                            vis.visit_span(inject_use_span);
                        }
                        ModKind::Unloaded => {}
                    }
                }
                ItemKind::ForeignMod(nm) => vis.visit_foreign_mod(nm),
                ItemKind::GlobalAsm(asm) => vis.visit_inline_asm(asm),
                ItemKind::TyAlias(box TyAlias {
                    defaultness,
                    generics,
                    where_clauses,
                    bounds,
                    ty,
                }) => {
                    visit_defaultness(vis, defaultness);
                    vis.visit_generics(generics);
                    visit_bounds(vis, bounds, BoundKind::Bound);
                    visit_opt(ty, |ty| vis.visit_ty(ty));
                    walk_ty_alias_where_clauses(vis, where_clauses);
                }
                ItemKind::Enum(enum_def, generics) => {
                    vis.visit_generics(generics);
                    vis.visit_enum_def(enum_def);
                }
                ItemKind::Struct(variant_data, generics)
                | ItemKind::Union(variant_data, generics) => {
                    vis.visit_generics(generics);
                    vis.visit_variant_data(variant_data);
                }
                ItemKind::Impl(box Impl {
                    defaultness,
                    safety,
                    generics,
                    constness,
                    polarity,
                    of_trait,
                    self_ty,
                    items,
                }) => {
                    visit_defaultness(vis, defaultness);
                    visit_safety(vis, safety);
                    vis.visit_generics(generics);
                    visit_constness(vis, constness);
                    visit_polarity(vis, polarity);
                    visit_opt(of_trait, |trait_ref| vis.visit_trait_ref(trait_ref));
                    vis.visit_ty(self_ty);
                    items.flat_map_in_place(|item| vis.flat_map_assoc_item(item, AssocCtxt::Impl));
                }
                ItemKind::Trait(box Trait { safety, is_auto: _, generics, bounds, items }) => {
                    visit_safety(vis, safety);
                    vis.visit_generics(generics);
                    visit_bounds(vis, bounds, BoundKind::Bound);
                    items.flat_map_in_place(|item| vis.flat_map_assoc_item(item, AssocCtxt::Trait));
                }
                ItemKind::TraitAlias(generics, bounds) => {
                    vis.visit_generics(generics);
                    visit_bounds(vis, bounds, BoundKind::Bound);
                }
                ItemKind::MacCall(m) => vis.visit_mac_call(m),
                ItemKind::MacroDef(def) => vis.visit_macro_def(def),
                ItemKind::Delegation(box Delegation {
                    id,
                    qself,
                    path,
                    rename,
                    body,
                    from_glob: _,
                }) => {
                    vis.visit_id(id);
                    vis.visit_qself(qself);
                    vis.visit_path(path);
                    if let Some(rename) = rename {
                        vis.visit_ident(rename);
                    }
                    if let Some(body) = body {
                        vis.visit_block(body);
                    }
                }
                ItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                    vis.visit_qself(qself);
                    vis.visit_path(prefix);
                    if let Some(suffixes) = suffixes {
                        for (ident, rename) in suffixes {
                            vis.visit_ident(ident);
                            if let Some(rename) = rename {
                                vis.visit_ident(rename);
                            }
                        }
                    }
                    if let Some(body) = body {
                        vis.visit_block(body);
                    }
                }
            }
        }
    }

    impl WalkItemKind for AssocItemKind {
        fn walk(&mut self, span: Span, id: NodeId, visitor: &mut impl MutVisitor) {
            match self {
                AssocItemKind::Const(item) => {
                    visit_const_item(item, visitor);
                }
                AssocItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
                    visit_defaultness(visitor, defaultness);
                    visitor.visit_fn(FnKind::Fn(sig, generics, body), span, id);
                }
                AssocItemKind::Type(box TyAlias {
                    defaultness,
                    generics,
                    where_clauses,
                    bounds,
                    ty,
                }) => {
                    visit_defaultness(visitor, defaultness);
                    visitor.visit_generics(generics);
                    visit_bounds(visitor, bounds, BoundKind::Bound);
                    visit_opt(ty, |ty| visitor.visit_ty(ty));
                    walk_ty_alias_where_clauses(visitor, where_clauses);
                }
                AssocItemKind::MacCall(mac) => visitor.visit_mac_call(mac),
                AssocItemKind::Delegation(box Delegation {
                    id,
                    qself,
                    path,
                    rename,
                    body,
                    from_glob: _,
                }) => {
                    visitor.visit_id(id);
                    visitor.visit_qself(qself);
                    visitor.visit_path(path);
                    if let Some(rename) = rename {
                        visitor.visit_ident(rename);
                    }
                    if let Some(body) = body {
                        visitor.visit_block(body);
                    }
                }
                AssocItemKind::DelegationMac(box DelegationMac {
                    qself,
                    prefix,
                    suffixes,
                    body,
                }) => {
                    visitor.visit_qself(qself);
                    visitor.visit_path(prefix);
                    if let Some(suffixes) = suffixes {
                        for (ident, rename) in suffixes {
                            visitor.visit_ident(ident);
                            if let Some(rename) = rename {
                                visitor.visit_ident(rename);
                            }
                        }
                    }
                    if let Some(body) = body {
                        visitor.visit_block(body);
                    }
                }
            }
        }
    }

    fn visit_const_item<T: MutVisitor>(
        ConstItem { defaultness, generics, ty, expr }: &mut ConstItem,
        visitor: &mut T,
    ) {
        visit_defaultness(visitor, defaultness);
        visitor.visit_generics(generics);
        visitor.visit_ty(ty);
        visit_opt(expr, |expr| visitor.visit_expr(expr));
    }

    fn walk_fn_header<T: MutVisitor>(vis: &mut T, header: &mut FnHeader) {
        let FnHeader { safety, coroutine_kind, constness, ext: _ } = header;
        visit_constness(vis, constness);
        coroutine_kind.as_mut().map(|coroutine_kind| vis.visit_coroutine_kind(coroutine_kind));
        visit_safety(vis, safety);
    }

    pub fn walk_crate<T: MutVisitor>(vis: &mut T, krate: &mut Crate) {
        let Crate { attrs, items, spans, id, is_placeholder: _ } = krate;
        vis.visit_id(id);
        visit_attrs(vis, attrs);
        items.flat_map_in_place(|item| vis.flat_map_item(item));
        let ModSpans { inner_span, inject_use_span } = spans;
        vis.visit_span(inner_span);
        vis.visit_span(inject_use_span);
    }

    /// Mutates one item, returning the item again.
    pub fn walk_flat_map_item<K: WalkItemKind>(
        visitor: &mut impl MutVisitor,
        mut item: P<Item<K>>,
    ) -> SmallVec<[P<Item<K>>; 1]> {
        let Item { ident, attrs, id, kind, vis, span, tokens } = item.deref_mut();
        visitor.visit_id(id);
        visit_attrs(visitor, attrs);
        visitor.visit_vis(vis);
        visitor.visit_ident(ident);
        kind.walk(*span, *id, visitor);
        visit_lazy_tts(visitor, tokens);
        visitor.visit_span(span);
        smallvec![item]
    }

    impl WalkItemKind for ForeignItemKind {
        fn walk(&mut self, span: Span, id: NodeId, visitor: &mut impl MutVisitor) {
            match self {
                ForeignItemKind::Static(box StaticItem { ty, mutability: _, expr, safety: _ }) => {
                    visitor.visit_ty(ty);
                    visit_opt(expr, |expr| visitor.visit_expr(expr));
                }
                ForeignItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
                    visit_defaultness(visitor, defaultness);
                    visitor.visit_fn(FnKind::Fn(sig, generics, body), span, id);
                }
                ForeignItemKind::TyAlias(box TyAlias {
                    defaultness,
                    generics,
                    where_clauses,
                    bounds,
                    ty,
                }) => {
                    visit_defaultness(visitor, defaultness);
                    visitor.visit_generics(generics);
                    visit_bounds(visitor, bounds, BoundKind::Bound);
                    visit_opt(ty, |ty| visitor.visit_ty(ty));
                    walk_ty_alias_where_clauses(visitor, where_clauses);
                }
                ForeignItemKind::MacCall(mac) => visitor.visit_mac_call(mac),
            }
        }
    }

    pub fn walk_pat<T: MutVisitor>(vis: &mut T, pat: &mut P<Pat>) {
        let Pat { id, kind, span, tokens } = pat.deref_mut();
        vis.visit_id(id);
        match kind {
            PatKind::Err(_guar) => {}
            PatKind::Wild | PatKind::Rest | PatKind::Never => {}
            PatKind::Ident(_binding_mode, ident, sub) => {
                vis.visit_ident(ident);
                visit_opt(sub, |sub| vis.visit_pat(sub));
            }
            PatKind::Lit(e) => vis.visit_expr(e),
            PatKind::TupleStruct(qself, path, elems) => {
                vis.visit_qself(qself);
                vis.visit_path(path);
                visit_thin_vec(elems, |elem| vis.visit_pat(elem));
            }
            PatKind::Path(qself, path) => {
                vis.visit_qself(qself);
                vis.visit_path(path);
            }
            PatKind::Struct(qself, path, fields, _etc) => {
                vis.visit_qself(qself);
                vis.visit_path(path);
                fields.flat_map_in_place(|field| vis.flat_map_pat_field(field));
            }
            PatKind::Box(inner) => vis.visit_pat(inner),
            PatKind::Deref(inner) => vis.visit_pat(inner),
            PatKind::Ref(inner, _mutbl) => vis.visit_pat(inner),
            PatKind::Range(e1, e2, Spanned { span: _, node: _ }) => {
                visit_opt(e1, |e| vis.visit_expr(e));
                visit_opt(e2, |e| vis.visit_expr(e));
                vis.visit_span(span);
            }
            PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
                visit_thin_vec(elems, |elem| vis.visit_pat(elem))
            }
            PatKind::Paren(inner) => vis.visit_pat(inner),
            PatKind::MacCall(mac) => vis.visit_mac_call(mac),
        }
        visit_lazy_tts(vis, tokens);
        vis.visit_span(span);
    }

    fn walk_inline_asm<T: MutVisitor>(vis: &mut T, asm: &mut InlineAsm) {
        // FIXME: Visit spans inside all this currently ignored stuff.
        let InlineAsm {
            asm_macro: _,
            template: _,
            template_strs: _,
            operands,
            clobber_abis: _,
            options: _,
            line_spans: _,
        } = asm;
        for (op, span) in operands {
            match op {
                InlineAsmOperand::In { expr, reg: _ }
                | InlineAsmOperand::Out { expr: Some(expr), reg: _, late: _ }
                | InlineAsmOperand::InOut { expr, reg: _, late: _ } => vis.visit_expr(expr),
                InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
                InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                    vis.visit_expr(in_expr);
                    if let Some(out_expr) = out_expr {
                        vis.visit_expr(out_expr);
                    }
                }
                InlineAsmOperand::Const { anon_const } => vis.visit_anon_const(anon_const),
                InlineAsmOperand::Sym { sym } => vis.visit_inline_asm_sym(sym),
                InlineAsmOperand::Label { block } => vis.visit_block(block),
            }
            vis.visit_span(span);
        }
    }

    fn walk_inline_asm_sym<T: MutVisitor>(
        vis: &mut T,
        InlineAsmSym { id, qself, path }: &mut InlineAsmSym,
    ) {
        vis.visit_id(id);
        vis.visit_qself(qself);
        vis.visit_path(path);
    }

    pub fn walk_expr<T: MutVisitor>(
        vis: &mut T,
        Expr { kind, id, span, attrs, tokens }: &mut Expr,
    ) {
        vis.visit_id(id);
        visit_attrs(vis, attrs);
        match kind {
            ExprKind::Array(exprs) => visit_thin_exprs(vis, exprs),
            ExprKind::ConstBlock(anon_const) => {
                vis.visit_anon_const(anon_const);
            }
            ExprKind::Repeat(expr, count) => {
                vis.visit_expr(expr);
                vis.visit_anon_const(count);
            }
            ExprKind::Tup(exprs) => visit_thin_exprs(vis, exprs),
            ExprKind::Call(f, args) => {
                vis.visit_expr(f);
                visit_thin_exprs(vis, args);
            }
            ExprKind::MethodCall(box MethodCall {
                seg: PathSegment { ident, id, args: seg_args },
                receiver,
                args: call_args,
                span,
            }) => {
                vis.visit_method_receiver_expr(receiver);
                vis.visit_id(id);
                vis.visit_ident(ident);
                visit_opt(seg_args, |args| vis.visit_generic_args(args));
                visit_thin_exprs(vis, call_args);
                vis.visit_span(span);
            }
            ExprKind::Binary(_binop, lhs, rhs) => {
                vis.visit_expr(lhs);
                vis.visit_expr(rhs);
            }
            ExprKind::Unary(_unop, ohs) => vis.visit_expr(ohs),
            ExprKind::Cast(expr, ty) => {
                vis.visit_expr(expr);
                vis.visit_ty(ty);
            }
            ExprKind::Type(expr, ty) => {
                vis.visit_expr(expr);
                vis.visit_ty(ty);
            }
            ExprKind::AddrOf(_kind, _mut, ohs) => vis.visit_expr(ohs),
            ExprKind::Let(pat, scrutinee, span, _recovered) => {
                vis.visit_pat(pat);
                vis.visit_expr(scrutinee);
                vis.visit_span(span);
            }
            ExprKind::If(cond, tr, fl) => {
                vis.visit_expr(cond);
                vis.visit_block(tr);
                visit_opt(fl, |fl| ensure_sufficient_stack(|| vis.visit_expr(fl)));
            }
            ExprKind::While(cond, body, label) => {
                visit_opt(label, |label| vis.visit_label(label));
                vis.visit_expr(cond);
                vis.visit_block(body);
            }
            ExprKind::ForLoop { pat, iter, body, label, kind: _ } => {
                visit_opt(label, |label| vis.visit_label(label));
                vis.visit_pat(pat);
                vis.visit_expr(iter);
                vis.visit_block(body);
            }
            ExprKind::Loop(body, label, span) => {
                visit_opt(label, |label| vis.visit_label(label));
                vis.visit_block(body);
                vis.visit_span(span);
            }
            ExprKind::Match(expr, arms, _kind) => {
                vis.visit_expr(expr);
                arms.flat_map_in_place(|arm| vis.flat_map_arm(arm));
            }
            ExprKind::Closure(box Closure {
                binder,
                capture_clause,
                constness,
                coroutine_kind,
                movability: _,
                fn_decl,
                body,
                fn_decl_span,
                fn_arg_span,
            }) => {
                visit_constness(vis, constness);
                coroutine_kind
                    .as_mut()
                    .map(|coroutine_kind| vis.visit_coroutine_kind(coroutine_kind));
                vis.visit_capture_by(capture_clause);
                vis.visit_fn(FnKind::Closure(binder, fn_decl, body), *span, *id);
                vis.visit_span(fn_decl_span);
                vis.visit_span(fn_arg_span);
            }
            ExprKind::Block(blk, label) => {
                visit_opt(label, |label| vis.visit_label(label));
                vis.visit_block(blk);
            }
            ExprKind::Gen(_capture_by, body, _kind, decl_span) => {
                vis.visit_block(body);
                vis.visit_span(decl_span);
            }
            ExprKind::Await(expr, await_kw_span) => {
                vis.visit_expr(expr);
                vis.visit_span(await_kw_span);
            }
            ExprKind::Assign(el, er, span) => {
                vis.visit_expr(el);
                vis.visit_expr(er);
                vis.visit_span(span);
            }
            ExprKind::AssignOp(_op, el, er) => {
                vis.visit_expr(el);
                vis.visit_expr(er);
            }
            ExprKind::Field(el, ident) => {
                vis.visit_expr(el);
                vis.visit_ident(ident);
            }
            ExprKind::Index(el, er, brackets_span) => {
                vis.visit_expr(el);
                vis.visit_expr(er);
                vis.visit_span(brackets_span);
            }
            ExprKind::Range(e1, e2, _lim) => {
                visit_opt(e1, |e1| vis.visit_expr(e1));
                visit_opt(e2, |e2| vis.visit_expr(e2));
            }
            ExprKind::Underscore => {}
            ExprKind::Path(qself, path) => {
                vis.visit_qself(qself);
                vis.visit_path(path);
            }
            ExprKind::Break(label, expr) => {
                visit_opt(label, |label| vis.visit_label(label));
                visit_opt(expr, |expr| vis.visit_expr(expr));
            }
            ExprKind::Continue(label) => {
                visit_opt(label, |label| vis.visit_label(label));
            }
            ExprKind::Ret(expr) => {
                visit_opt(expr, |expr| vis.visit_expr(expr));
            }
            ExprKind::Yeet(expr) => {
                visit_opt(expr, |expr| vis.visit_expr(expr));
            }
            ExprKind::Become(expr) => vis.visit_expr(expr),
            ExprKind::InlineAsm(asm) => vis.visit_inline_asm(asm),
            ExprKind::FormatArgs(fmt) => vis.visit_format_args(fmt),
            ExprKind::OffsetOf(container, fields) => {
                vis.visit_ty(container);
                for field in fields.iter_mut() {
                    vis.visit_ident(field);
                }
            }
            ExprKind::MacCall(mac) => vis.visit_mac_call(mac),
            ExprKind::Struct(se) => {
                let StructExpr { qself, path, fields, rest } = se.deref_mut();
                vis.visit_qself(qself);
                vis.visit_path(path);
                fields.flat_map_in_place(|field| vis.flat_map_expr_field(field));
                match rest {
                    StructRest::Base(expr) => vis.visit_expr(expr),
                    StructRest::Rest(_span) => {}
                    StructRest::None => {}
                }
            }
            ExprKind::Paren(expr) => {
                vis.visit_expr(expr);
            }
            ExprKind::Yield(expr) => {
                visit_opt(expr, |expr| vis.visit_expr(expr));
            }
            ExprKind::Try(expr) => vis.visit_expr(expr),
            ExprKind::TryBlock(body) => vis.visit_block(body),
            ExprKind::Lit(_token) => {}
            ExprKind::IncludedBytes(_bytes) => {}
            ExprKind::Err(_guar) => {}
            ExprKind::Dummy => {}
        }
        visit_lazy_tts(vis, tokens);
        vis.visit_span(span);
    }

    pub fn noop_filter_map_expr<T: MutVisitor>(vis: &mut T, mut e: P<Expr>) -> Option<P<Expr>> {
        Some({
            vis.visit_expr(&mut e);
            e
        })
    }

    pub fn walk_flat_map_stmt<T: MutVisitor>(
        vis: &mut T,
        Stmt { kind, mut span, mut id }: Stmt,
    ) -> SmallVec<[Stmt; 1]> {
        vis.visit_id(&mut id);
        let stmts: SmallVec<_> = walk_flat_map_stmt_kind(vis, kind)
            .into_iter()
            .map(|kind| Stmt { id, kind, span })
            .collect();
        if stmts.len() > 1 {
            panic!(
                "cloning statement `NodeId`s is prohibited by default, \
                 the visitor should implement custom statement visiting"
            );
        }
        vis.visit_span(&mut span);
        stmts
    }

    fn walk_flat_map_stmt_kind<T: MutVisitor>(
        vis: &mut T,
        kind: StmtKind,
    ) -> SmallVec<[StmtKind; 1]> {
        match kind {
            StmtKind::Let(mut local) => smallvec![StmtKind::Let({
                vis.visit_local(&mut local);
                local
            })],
            StmtKind::Item(item) => {
                vis.flat_map_item(item).into_iter().map(StmtKind::Item).collect()
            }
            StmtKind::Expr(expr) => {
                vis.filter_map_expr(expr).into_iter().map(StmtKind::Expr).collect()
            }
            StmtKind::Semi(expr) => {
                vis.filter_map_expr(expr).into_iter().map(StmtKind::Semi).collect()
            }
            StmtKind::Empty => smallvec![StmtKind::Empty],
            StmtKind::MacCall(mut mac) => {
                let MacCallStmt { mac: mac_, style: _, attrs, tokens } = mac.deref_mut();
                visit_attrs(vis, attrs);
                vis.visit_mac_call(mac_);
                visit_lazy_tts(vis, tokens);
                smallvec![StmtKind::MacCall(mac)]
            }
        }
    }

    fn walk_vis<T: MutVisitor>(vis: &mut T, visibility: &mut Visibility) {
        let Visibility { kind, span, tokens } = visibility;
        match kind {
            VisibilityKind::Public | VisibilityKind::Inherited => {}
            VisibilityKind::Restricted { path, id, shorthand: _ } => {
                vis.visit_id(id);
                vis.visit_path(path);
            }
        }
        visit_lazy_tts(vis, tokens);
        vis.visit_span(span);
    }

    /// Some value for the AST node that is valid but possibly meaningless. Similar
    /// to `Default` but not intended for wide use. The value will never be used
    /// meaningfully, it exists just to support unwinding in `visit_clobber` in the
    /// case where its closure panics.
    pub trait DummyAstNode {
        fn dummy() -> Self;
    }

    impl<T> DummyAstNode for Option<T> {
        fn dummy() -> Self {
            Default::default()
        }
    }

    impl<T: DummyAstNode + 'static> DummyAstNode for P<T> {
        fn dummy() -> Self {
            P(DummyAstNode::dummy())
        }
    }

    impl DummyAstNode for Item {
        fn dummy() -> Self {
            Item {
                attrs: Default::default(),
                id: DUMMY_NODE_ID,
                span: Default::default(),
                vis: Visibility {
                    kind: VisibilityKind::Public,
                    span: Default::default(),
                    tokens: Default::default(),
                },
                ident: Ident::empty(),
                kind: ItemKind::ExternCrate(None),
                tokens: Default::default(),
            }
        }
    }

    impl DummyAstNode for Expr {
        fn dummy() -> Self {
            Expr {
                id: DUMMY_NODE_ID,
                kind: ExprKind::Dummy,
                span: Default::default(),
                attrs: Default::default(),
                tokens: Default::default(),
            }
        }
    }

    impl DummyAstNode for Ty {
        fn dummy() -> Self {
            Ty {
                id: DUMMY_NODE_ID,
                kind: TyKind::Dummy,
                span: Default::default(),
                tokens: Default::default(),
            }
        }
    }

    impl DummyAstNode for Pat {
        fn dummy() -> Self {
            Pat {
                id: DUMMY_NODE_ID,
                kind: PatKind::Wild,
                span: Default::default(),
                tokens: Default::default(),
            }
        }
    }

    impl DummyAstNode for Stmt {
        fn dummy() -> Self {
            Stmt { id: DUMMY_NODE_ID, kind: StmtKind::Empty, span: Default::default() }
        }
    }

    impl DummyAstNode for Crate {
        fn dummy() -> Self {
            Crate {
                attrs: Default::default(),
                items: Default::default(),
                spans: Default::default(),
                id: DUMMY_NODE_ID,
                is_placeholder: Default::default(),
            }
        }
    }

    impl<N: DummyAstNode, T: DummyAstNode> DummyAstNode for crate::ast_traits::AstNodeWrapper<N, T> {
        fn dummy() -> Self {
            crate::ast_traits::AstNodeWrapper::new(N::dummy(), T::dummy())
        }
    }

    #[derive(Debug)]
    pub enum FnKind<'a> {
        /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
        Fn(&'a mut FnSig, &'a mut Generics, &'a mut Option<P<Block>>),

        /// E.g., `|x, y| body`.
        Closure(&'a mut ClosureBinder, &'a mut P<FnDecl>, &'a mut P<Expr>),
    }
}
