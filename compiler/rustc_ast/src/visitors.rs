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
            (
                $ty: ty
                $$(, $$($arg: ident)? $$(_ $ignored_arg: ident)?: $arg_ty: ty)*;
                $visit: ident, $walk: ident,
                $flat_map: ident, $walk_flat_map: ident
            ) => {
                make_visit!{
                    $ty
                    $$(, $$($arg)? $$(_ $ignored_arg)?: $arg_ty)*;
                    $visit, $walk
                }

                macro_if!{$($mut)? {
                    fn $flat_map(
                        &mut self,
                        node: $ty
                        $$(, $$($arg)? $$($ignored_arg)?: $arg_ty)*
                    ) -> SmallVec<[$ty; 1]> {
                        $walk_flat_map(self, node $$(, $$($arg)? $$($ignored_arg)?)*)
                    }
                }}
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

        macro_rules! mut_only_visit {
            ($name: ident) => {
                macro_rules! $name {
                    ($vis: expr, $arg: expr) => {
                        macro_if!{ $($mut)? {
                            $name($vis, $arg)
                        } else {
                            // assign to _ to prevent unused_variable warnings
                            {let _ = (&$vis, &$arg);}
                        }}
                    };
                }
            };
        }

        mut_only_visit!{visit_lazy_tts}
        mut_only_visit!{visit_delim_args}

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

        macro_rules! fn_kind_derives {
            ($i: item) => {
                macro_if!{$($mut)? {
                    #[derive(Debug)]
                    $i
                } else {
                    #[derive(Debug, Copy, Clone)]
                    $i
                }}
            }
        }

        fn_kind_derives!{
            pub enum FnKind<'a> {
                /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
                Fn(FnCtxt, &'a $($mut)? Ident, &'a $($mut)? FnSig, &'a $($mut)? Visibility, &'a $($mut)? Generics, &'a $($mut)? Option<P<Block>>),

                /// E.g., `|x, y| body`.
                Closure(&'a $($mut)? ClosureBinder, &'a $($mut)? Option<CoroutineKind>, &'a $($mut)? P!(FnDecl), &'a $($mut)? P!(Expr)),
            }
        }

        macro_rules! FnKind {
            () => {
                macro_if!{$($lt)? { FnKind<$($lt)?> } else { FnKind<'_> }}
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


            // Methods in these traits have the form:
            //
            //   fn visit_t(&mut self, t: ref_t!(T)) -> result!();
            //
            // In addition to those the mutable version also has methods of the form:
            //
            //   fn flat_map_t(&mut self, t: T) -> SmallVec<[T; 1]>;
            //   fn filter_map_t(&mut self, t: T) -> Option<T>;
            //
            // Any additions to this trait should happen in form of a call to a public
            // `walk_*` function that only calls out to the visitor again, not other
            // `walk_*` functions. This is a necessary API workaround to the problem of
            // not being able to call out to the default implementation of an overridden
            // method.
            //
            // When writing these methods, it is better to use destructuring like this:
            //
            //   fn walk_abc(&mut self, abc: &mut ABC) {
            //       let ABC { a, b, c: _ } = abc;
            //       visit_a(a);
            //       visit_b(b);
            //   }
            //
            // than to use field access like this:
            //
            //   fn walk_abc(&mut self, abc: &mut ABC) {
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


            macro_if!{$($mut)? {
                /// Mutable token visiting only exists for the `macro_rules` token marker and should not be
                /// used otherwise. Token visitor would be entirely separate from the regular visitor if
                /// the marker didn't have to visit AST fragments in nonterminal tokens.
                const VISIT_TOKENS: bool = false;

                make_visit!{MetaItem; visit_meta_item, walk_meta_item}
                make_visit!{MetaItemInner; visit_meta_list_item, walk_meta_list_item}

                fn flat_map_stmt(&mut self, s: Stmt) -> SmallVec<[Stmt; 1]> {
                    walk_flat_map_stmt(self, s)
                }

                fn filter_map_expr(&mut self, e: P<Expr>) -> Option<P<Expr>> {
                    noop_filter_map_expr(self, e)
                }

                fn visit_id(&mut self, _id: &mut NodeId) {
                    // Do nothing.
                }

                fn visit_span(&mut self, _sp: &mut Span) {
                    // Do nothing.
                }
            } else {
                /// The result type of the `visit_*` methods. Can be either `()`,
                /// or `ControlFlow<T>`.
                type Result: VisitorResult = ();

                make_visit!{Stmt; visit_stmt, walk_stmt}
            }}

            make_visit!{AngleBracketedArgs; visit_angle_bracketed_parameter_data, walk_angle_bracketed_parameter_data}
            make_visit!{AnonConst; visit_anon_const, walk_anon_const}
            make_visit!{Arm; visit_arm, walk_arm, flat_map_arm, walk_flat_map_arm}
            make_visit!{AssocItemConstraint; visit_assoc_item_constraint, walk_assoc_item_constraint}
            make_visit!{AttrArgs; visit_attr_args, walk_attr_args}
            make_visit!{Attribute; visit_attribute, walk_attribute}
            make_visit!{Block; visit_block, walk_block}
            make_visit!{CaptureBy; visit_capture_by, walk_capture_by}
            make_visit!{ClosureBinder; visit_closure_binder, walk_closure_binder}
            make_visit!{Const; visit_constness, walk_constness}
            make_visit!{CoroutineKind; visit_coroutine_kind, walk_coroutine_kind}
            make_visit!{Crate; visit_crate, walk_crate}
            make_visit!{Defaultness; visit_defaultness, walk_defaultness}
            make_visit!{EnumDef; visit_enum_def, walk_enum_def}
            make_visit!{ExprField; visit_expr_field, walk_expr_field, flat_map_expr_field, walk_flat_map_expr_field}
            make_visit!{FieldDef; visit_field_def, walk_field_def, flat_map_field_def, walk_flat_map_field_def}
            make_visit!{FnDecl; visit_fn_decl, walk_fn_decl}
            make_visit!{FnHeader; visit_fn_header, walk_fn_header}
            make_visit!{FnRetTy; visit_fn_ret_ty, walk_fn_ret_ty}
            make_visit!{ForeignMod; visit_foreign_mod, walk_foreign_mod}
            make_visit!{FormatArgs; visit_format_args, walk_format_args}
            make_visit!{GenericArg; visit_generic_arg, walk_generic_arg}
            make_visit!{GenericArgs; visit_generic_args, walk_generic_args}
            make_visit!{GenericBound, _ ctxt: BoundKind; visit_param_bound, walk_param_bound}
            make_visit!{GenericParam; visit_generic_param, walk_generic_param, flat_map_generic_param, walk_flat_map_generic_param}
            make_visit!{Generics; visit_generics, walk_generics}
            make_visit!{Ident; visit_ident, walk_ident}
            make_visit!{ImplPolarity; visit_impl_polarity, walk_impl_polarity}
            make_visit!{InlineAsm; visit_inline_asm, walk_inline_asm}
            make_visit!{InlineAsmSym; visit_inline_asm_sym, walk_inline_asm_sym}
            make_visit!{Label; visit_label, walk_label}
            make_visit!{Lifetime, _ ctxt: LifetimeCtxt; visit_lifetime, walk_lifetime}
            make_visit!{Local; visit_local, walk_local}
            make_visit!{MacCall; visit_mac_call, walk_mac_call}
            make_visit!{MacroDef, _ id: NodeId; visit_macro_def, walk_macro_def}
            make_visit!{MutTy; visit_mt, walk_mt}
            make_visit!{Option<P<QSelf>>; visit_qself, walk_qself}
            make_visit!{Param; visit_param, walk_param, flat_map_param, walk_flat_map_param}
            make_visit!{ParenthesizedArgs; visit_parenthesized_parameter_data, walk_parenthesized_parameter_data}
            make_visit!{PatField; visit_pat_field, walk_pat_field, flat_map_pat_field, walk_flat_map_pat_field}
            make_visit!{Path, _ id: NodeId; visit_path, walk_path}
            make_visit!{PathSegment; visit_path_segment, walk_path_segment}
            make_visit!{PolyTraitRef; visit_poly_trait_ref, walk_poly_trait_ref}
            make_visit!{PreciseCapturingArg; visit_precise_capturing_arg, walk_precise_capturing_arg}
            make_visit!{Safety; visit_safety, walk_safety}
            make_visit!{TraitRef; visit_trait_ref, walk_trait_ref}
            make_visit!{TyAliasWhereClauses; visit_ty_alias_where_clauses, walk_ty_alias_where_clauses}
            make_visit!{UseTree, id: NodeId, _ nested: bool; visit_use_tree, walk_use_tree}
            make_visit!{Variant; visit_variant, walk_variant, flat_map_variant, walk_flat_map_variant}
            make_visit!{VariantData; visit_variant_data, walk_variant_data}
            make_visit!{Visibility; visit_vis, walk_vis}
            make_visit!{WhereClause; visit_where_clause, walk_where_clause}
            make_visit!{WherePredicate; visit_where_predicate, walk_where_predicate}

            // FIXME: Remove these P!s
            make_visit!{P!(Expr); visit_expr, walk_expr}
            make_visit!{P!(Pat); visit_pat, walk_pat}
            make_visit!{P!(Ty); visit_ty, walk_ty}

            // Item variants
            // FIXME: Remove these P!s
            make_visit!{P!(Item); visit_item, walk_item, flat_map_item, walk_flat_map_item}
            make_visit!{P!(AssocItem), ctxt: AssocCtxt; visit_assoc_item, walk_assoc_item, flat_map_assoc_item, walk_flat_map_assoc_item}
            make_visit!{P!(ForeignItem); visit_foreign_item, walk_item, flat_map_foreign_item, walk_flat_map_foreign_item}

            fn visit_fn(&mut self, fn_kind: FnKind!(), _span: Span, _id: NodeId) -> result!() {
                walk_fn(self, fn_kind)
            }

            /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
            /// It can be removed once that feature is stabilized.
            fn visit_method_receiver_expr(&mut self, ex: ref_t!(P!(Expr))) -> result!() {
                self.visit_expr(ex)
            }

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

        pub fn walk_assoc_item_constraint<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            constraint: ref_t!(AssocItemConstraint)
        ) -> result!(V) {
            let AssocItemConstraint { id, ident, gen_args, kind, span } = constraint;
            try_v!(visit_id!(vis, id));
            try_v!(vis.visit_ident(ident));
            visit_o!(gen_args, |gen_args| vis.visit_generic_args(gen_args));
            match kind {
                AssocItemConstraintKind::Equality { term } => {
                    match term {
                        Term::Ty(ty) => {
                            try_v!(vis.visit_ty(ty));
                        }
                        Term::Const(c) => {
                            try_v!(vis.visit_anon_const(c));
                        }
                    }
                }
                AssocItemConstraintKind::Bound { bounds } => {
                    visit_list!(vis, visit_param_bound, bounds; BoundKind::Bound);
                }
            }
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_attr_args<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            args: ref_t!(AttrArgs)
        ) -> result!(V) {
            match args {
                AttrArgs::Empty => {}
                AttrArgs::Delimited(args) => {
                    visit_delim_args!(vis, args);
                }
                AttrArgs::Eq(eq_span, AttrArgsEq::Ast(expr)) => {
                    try_v!(vis.visit_expr(expr));
                    try_v!(visit_span!(vis, eq_span));
                }
                AttrArgs::Eq(_eq_span, AttrArgsEq::Hir(lit)) => {
                    unreachable!("in literal form when visiting mac args eq: {:?}", lit)
                }
            }
            return_result!(V)
        }

        pub fn walk_attribute<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            attr: ref_t!(Attribute)
        ) -> result!(V) {
            let Attribute { kind, id:_, style: _, span } = attr;
            match kind {
                AttrKind::Normal(normal) => {
                    let NormalAttr {
                        item: AttrItem { unsafety, path, args, tokens },
                        tokens: attr_tokens,
                    } = &$($mut)? **normal;
                    try_v!(vis.visit_safety(unsafety));
                    try_v!(vis.visit_path(path, DUMMY_NODE_ID));
                    try_v!(vis.visit_attr_args(args));
                    visit_lazy_tts!(vis, tokens);
                    visit_lazy_tts!(vis, attr_tokens);
                }
                AttrKind::DocComment(_kind, _sym) => {}
            }
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_block<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            block: ref_t!(Block)
        ) -> result!(V) {
            let Block { id, stmts, rules: _, span, tokens, could_be_bare_literal: _ } = block;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_stmt, flat_map_stmt, stmts);
            visit_lazy_tts!(vis, tokens);
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

        pub fn walk_constness<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            constness: ref_t!(Const)
        ) -> result!(V) {
            match constness {
                Const::Yes(span) => {
                    try_v!(visit_span!(vis, span));
                }
                Const::No => {}
            }
            return_result!(V)
        }

        pub fn walk_coroutine_kind<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            coroutine_kind: ref_t!(CoroutineKind)
        ) -> result!(V) {
            match coroutine_kind {
                CoroutineKind::Async { span, closure_id, return_impl_trait_id }
                | CoroutineKind::Gen { span, closure_id, return_impl_trait_id }
                | CoroutineKind::AsyncGen { span, closure_id, return_impl_trait_id } => {
                    try_v!(visit_id!(vis, closure_id));
                    try_v!(visit_id!(vis, return_impl_trait_id));
                    try_v!(visit_span!(vis, span));
                }
            }
            return_result!(V)
        }

        pub fn walk_crate<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            krate: ref_t!(Crate)
        ) -> result!(V) {
            let Crate { attrs, items, spans, id, is_placeholder: _ } = krate;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            visit_list!(vis, visit_item, flat_map_item, items);
            let ModSpans { inner_span, inject_use_span } = spans;
            try_v!(visit_span!(vis, inner_span));
            try_v!(visit_span!(vis, inject_use_span));
            return_result!(V)
        }

        pub fn walk_defaultness<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            defaultness: ref_t!(Defaultness)
        ) -> result!(V) {
            match defaultness {
                Defaultness::Default(span) => {
                    try_v!(visit_span!(vis, span))
                }
                Defaultness::Final => {}
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

        pub fn walk_field_def<$($lt,)? V: $trait$(<$lt>)?>(
            visitor: &mut V,
            fd: ref_t!(FieldDef)
        ) -> result!(V) {
            let FieldDef { span, ident, vis, id, ty, attrs, is_placeholder: _ } = fd;
            try_v!(visit_id!(visitor, id));
            visit_list!(visitor, visit_attribute, attrs);
            try_v!(visitor.visit_vis(vis));
            visit_o!(ident, |ident| visitor.visit_ident(ident));
            try_v!(visitor.visit_ty(ty));
            try_v!(visit_span!(visitor, span));
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

        pub fn walk_fn_header<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            header: ref_t!(FnHeader)
        ) -> result!(V) {
            let FnHeader { safety, coroutine_kind, constness, ext: _ } = header;
            try_v!(vis.visit_constness(constness));
            visit_o!(coroutine_kind, |ck| vis.visit_coroutine_kind(ck));
            try_v!(vis.visit_safety(safety));
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

        pub fn walk_foreign_mod<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            foreign_mod: ref_t!(ForeignMod)
        ) -> result!(V) {
            let ForeignMod { safety, abi: _, items } = foreign_mod;
            try_v!(vis.visit_safety(safety));
            visit_list!(vis, visit_foreign_item, flat_map_foreign_item, items);
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

        pub fn walk_generic_arg<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            arg: ref_t!(GenericArg)
        ) -> result!(V) {
            match arg {
                GenericArg::Lifetime(lt) => try_v!(vis.visit_lifetime(lt, LifetimeCtxt::GenericArg)),
                GenericArg::Type(ty) => try_v!(vis.visit_ty(ty)),
                GenericArg::Const(ct) => try_v!(vis.visit_anon_const(ct)),
            }
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

        pub fn walk_param_bound<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            bound: ref_t!(GenericBound)
        ) -> result!(V) {
            match bound {
                GenericBound::Trait(typ) => {
                    try_v!(vis.visit_poly_trait_ref(typ));
                }
                GenericBound::Outlives(lifetime) => {
                    try_v!(vis.visit_lifetime(lifetime, LifetimeCtxt::Bound));
                }
                GenericBound::Use(args, span) => {
                    visit_list!(vis, visit_precise_capturing_arg, args);
                    try_v!(visit_span!(vis, span))
                }
            }
            return_result!(V)
        }

        pub fn walk_generic_param<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            param: ref_t!(GenericParam)
        ) -> result!(V) {
            let GenericParam { id, ident, attrs, bounds, kind, colon_span, is_placeholder: _ } = param;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            try_v!(vis.visit_ident(ident));
            visit_list!(vis, visit_param_bound, bounds; BoundKind::Bound);
            match kind {
                GenericParamKind::Lifetime => {}
                GenericParamKind::Type { default } => {
                    visit_o!(default, |default| vis.visit_ty(default));
                }
                GenericParamKind::Const { ty, kw_span, default } => {
                    try_v!(vis.visit_ty(ty));
                    visit_o!(default, |default| vis.visit_anon_const(default));
                    try_v!(visit_span!(vis, kw_span));
                }
            }
            visit_o!(colon_span, |span| visit_span!(vis, span));
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

        pub fn walk_impl_polarity<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            polarity: ref_t!(ImplPolarity)
        ) -> result!(V) {
            match polarity {
                ImplPolarity::Positive => {}
                ImplPolarity::Negative(span) => {
                    try_v!(visit_span!(vis, span));
                }
            }
            return_result!(V)
        }

        pub fn walk_inline_asm<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            asm: ref_t!(InlineAsm)
        ) -> result!(V) {
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
                    | InlineAsmOperand::InOut { expr, reg: _, late: _ } => {
                        try_v!(vis.visit_expr(expr));
                    }
                    InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
                    InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                        try_v!(vis.visit_expr(in_expr));
                        visit_o!(out_expr, |out_expr| vis.visit_expr(out_expr));
                    }
                    InlineAsmOperand::Const { anon_const } => {
                        try_v!(vis.visit_anon_const(anon_const));
                    }
                    InlineAsmOperand::Sym { sym } => {
                        try_v!(vis.visit_inline_asm_sym(sym));
                    }
                    InlineAsmOperand::Label { block } => {
                        try_v!(vis.visit_block(block));
                    }
                }
                try_v!(visit_span!(vis, span));
            }
            return_result!(V)
        }

        pub fn walk_inline_asm_sym<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            asm_sym: ref_t!(InlineAsmSym)
        ) -> result!(V) {
            let InlineAsmSym { id, qself, path } = asm_sym;
            try_v!(visit_id!(vis, id));
            try_v!(vis.visit_qself(qself));
            try_v!(vis.visit_path(path, *id));
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

        pub fn walk_local<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            local: ref_t!(Local)
        ) -> result!(V) {
            let Local { id, pat, ty, kind, span, colon_sp, attrs, tokens } = local;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            try_v!(vis.visit_pat(pat));
            visit_o!(ty, |ty| vis.visit_ty(ty));
            match kind {
                LocalKind::Decl => {}
                LocalKind::Init(init) => {
                    try_v!(vis.visit_expr(init));
                }
                LocalKind::InitElse(init, els) => {
                    try_v!(vis.visit_expr(init));
                    try_v!(vis.visit_block(els));
                }
            }
            visit_lazy_tts!(vis, tokens);
            visit_o!(colon_sp, |sp| try_v!(visit_span!(vis, sp)));
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_mac_call<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            mac: ref_t!(MacCall)
        ) -> result!(V) {
            let MacCall { path, args } = mac;
            try_v!(vis.visit_path(path, DUMMY_NODE_ID));
            visit_delim_args!(vis, args);
            return_result!(V)
        }

        pub fn walk_macro_def<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            macro_def: ref_t!(MacroDef),
        ) -> result!(V) {
            let MacroDef { body, macro_rules: _ } = macro_def;
            visit_delim_args!(vis, body);
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

        pub fn walk_qself<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            qself: ref_t!(Option<P<QSelf>>)
        ) -> result!(V) {
            if let Some(qself) = qself {
                let QSelf { ty, path_span, position: _ } = &$($mut)? **qself;
                try_v!(vis.visit_ty(ty));
                try_v!(visit_span!(vis, path_span));
            }
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

        pub fn walk_path<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            path: ref_t!(Path)
        ) -> result!(V) {
            let Path { span, segments, tokens } = path;
            visit_list!(vis, visit_path_segment, segments);
            visit_lazy_tts!(vis, tokens);
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_path_segment<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            segment: ref_t!(PathSegment)
        ) -> result!(V) {
            let PathSegment { id, ident, args } = segment;
            try_v!(visit_id!(vis, id));
            try_v!(vis.visit_ident(ident));
            visit_o!(args, |args| vis.visit_generic_args(args));
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

        pub fn walk_precise_capturing_arg<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            arg: ref_t!(PreciseCapturingArg)
        ) -> result!(V) {
            match arg {
                PreciseCapturingArg::Lifetime(lt) => {
                    try_v!(vis.visit_lifetime(lt, LifetimeCtxt::GenericArg));
                }
                PreciseCapturingArg::Arg(path, id) => {
                    try_v!(visit_id!(vis, id));
                    try_v!(vis.visit_path(path, *id));
                }
            }
            return_result!(V)
        }

        pub fn walk_safety<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            safety: ref_t!(Safety)
        ) -> result!(V) {
            match safety {
                Safety::Unsafe(span)
                | Safety::Safe(span) => {
                    try_v!(visit_span!(vis, span))
                }
                Safety::Default => {}
            }
            return_result!(V)
        }

        pub fn walk_trait_ref<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            trait_ref: ref_t!(TraitRef)
        ) -> result!(V) {
            let TraitRef { path, ref_id } = trait_ref;
            try_v!(visit_id!(vis, ref_id));
            try_v!(vis.visit_path(path, *ref_id));
            return_result!(V)
        }

        pub fn walk_ty_alias_where_clauses<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            tawcs: ref_t!(TyAliasWhereClauses)
        ) -> result!(V) {
            let TyAliasWhereClauses { before, after, split: _ } = tawcs;
            let TyAliasWhereClause { has_where_token: _, span: span_before } = before;
            let TyAliasWhereClause { has_where_token: _, span: span_after } = after;
            try_v!(visit_span!(vis, span_before));
            try_v!(visit_span!(vis, span_after));
            return_result!(V)
        }

        pub fn walk_use_tree<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            use_tree: ref_t!(UseTree),
            id: NodeId,
        ) -> result!(V) {
            let UseTree { prefix, kind, span } = use_tree;
            try_v!(vis.visit_path(prefix, id));
            match kind {
                UseTreeKind::Simple(rename) => {
                    // The extra IDs are handled during AST lowering.
                    visit_o!(rename, |rename: ref_t!(Ident)| vis.visit_ident(rename));
                }
                UseTreeKind::Nested { items, span } => {
                    for (tree, id) in items {
                        try_v!(visit_id!(vis, id));
                        vis.visit_use_tree(tree, *id, true);
                    }
                    try_v!(visit_span!(vis, span));
                }
                UseTreeKind::Glob => {}
            }
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

        pub fn walk_vis<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            visibility: ref_t!(Visibility)
        ) -> result!(V) {
            let Visibility { kind, span, tokens } = visibility;
            match kind {
                VisibilityKind::Public | VisibilityKind::Inherited => {}
                VisibilityKind::Restricted { path, id, shorthand: _ } => {
                    try_v!(visit_id!(vis, id));
                    try_v!(vis.visit_path(path, *id));
                }
            }
            visit_lazy_tts!(vis, tokens);
            try_v!(visit_span!(vis, span));
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

        pub fn walk_expr<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            expr: ref_t!(Expr)
        ) -> result!(V) {
            let Expr { kind, id, span, attrs, tokens } = expr;
            try_v!(visit_id!(vis, id));
            visit_list!(vis, visit_attribute, attrs);
            match kind {
                ExprKind::Array(exprs)
                | ExprKind::Tup(exprs)  => {
                    visit_list!(vis, visit_expr, filter_map_expr, exprs);
                }
                ExprKind::ConstBlock(anon_const) => {
                    try_v!(vis.visit_anon_const(anon_const));
                }
                ExprKind::Repeat(expr, count) => {
                    try_v!(vis.visit_expr(expr));
                    try_v!(vis.visit_anon_const(count));
                }
                ExprKind::Call(f, args) => {
                    try_v!(vis.visit_expr(f));
                    visit_list!(vis, visit_expr, filter_map_expr, args);
                }
                ExprKind::MethodCall(box MethodCall {
                    seg,
                    receiver,
                    args,
                    span,
                }) => {
                    try_v!(vis.visit_method_receiver_expr(receiver));
                    try_v!(vis.visit_path_segment(seg));
                    visit_list!(vis, visit_expr, filter_map_expr, args);
                    try_v!(visit_span!(vis, span));
                }
                ExprKind::Binary(_op, lhs, rhs) => {
                    try_v!(vis.visit_expr(lhs));
                    try_v!(vis.visit_expr(rhs));
                }
                ExprKind::Unary(_op, expr) => {
                    try_v!(vis.visit_expr(expr));
                }
                ExprKind::Cast(expr, typ)
                | ExprKind::Type(expr, typ) => {
                    try_v!(vis.visit_expr(expr));
                    try_v!(vis.visit_ty(typ));
                }
                ExprKind::AddrOf(_kind, _mutbl, expr) => {
                    try_v!(vis.visit_expr(expr));
                }
                ExprKind::Let(pat, expr, span, _recovered) => {
                    try_v!(vis.visit_pat(pat));
                    try_v!(vis.visit_expr(expr));
                    try_v!(visit_span!(vis, span));
                }
                ExprKind::If(cond, if_block, else_block) => {
                    try_v!(vis.visit_expr(cond));
                    try_v!(vis.visit_block(if_block));
                    visit_o!(else_block, |else_block| ensure_sufficient_stack(|| vis.visit_expr(else_block)));
                }
                ExprKind::While(cond, body, label) => {
                    visit_o!(label, |label| vis.visit_label(label));
                    try_v!(vis.visit_expr(cond));
                    try_v!(vis.visit_block(body));
                }
                ExprKind::ForLoop { pat, iter, body, label, kind: _ } => {
                    visit_o!(label, |label| vis.visit_label(label));
                    try_v!(vis.visit_pat(pat));
                    try_v!(vis.visit_expr(iter));
                    try_v!(vis.visit_block(body));
                }
                ExprKind::Loop(body, label, span) => {
                    visit_o!(label, |label| vis.visit_label(label));
                    try_v!(vis.visit_block(body));
                    try_v!(visit_span!(vis, span));
                }
                ExprKind::Match(expr, arms, _kind) => {
                    try_v!(vis.visit_expr(expr));
                    visit_list!(vis, visit_arm, flat_map_arm, arms);
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
                    try_v!(vis.visit_constness(constness));
                    try_v!(vis.visit_capture_by(capture_clause));
                    try_v!(vis.visit_fn(FnKind::Closure(binder, coroutine_kind, fn_decl, body), *span, *id));
                    try_v!(visit_span!(vis, fn_decl_span));
                    try_v!(visit_span!(vis, fn_arg_span));
                }
                ExprKind::Block(block, label) => {
                    visit_o!(label, |label| vis.visit_label(label));
                    try_v!(vis.visit_block(block));
                }
                ExprKind::Gen(capture_by, body, _kind, decl_span) => {
                    try_v!(vis.visit_capture_by(capture_by));
                    try_v!(vis.visit_block(body));
                    try_v!(visit_span!(vis, decl_span));
                }
                ExprKind::Await(expr, await_kw_span) => {
                    try_v!(vis.visit_expr(expr));
                    try_v!(visit_span!(vis, await_kw_span));
                }
                ExprKind::Assign(lhs, rhs, span) => {
                    try_v!(vis.visit_expr(lhs));
                    try_v!(vis.visit_expr(rhs));
                    try_v!(visit_span!(vis, span));
                }
                ExprKind::AssignOp(_op, lhs, rhs) => {
                    try_v!(vis.visit_expr(lhs));
                    try_v!(vis.visit_expr(rhs));
                }
                ExprKind::Field(el, ident) => {
                    try_v!(vis.visit_expr(el));
                    try_v!(vis.visit_ident(ident));
                }
                ExprKind::Index(main_expr, index_expr, brackets_span) => {
                    try_v!(vis.visit_expr(main_expr));
                    try_v!(vis.visit_expr(index_expr));
                    try_v!(visit_span!(vis, brackets_span));
                }
                ExprKind::Range(start, end, _lim) => {
                    visit_o!(start, |start| vis.visit_expr(start));
                    visit_o!(end, |end| vis.visit_expr(end));
                }
                ExprKind::Underscore => {}
                ExprKind::Path(qself, path) => {
                    try_v!(vis.visit_qself(qself));
                    try_v!(vis.visit_path(path, *id));
                }
                ExprKind::Break(label, expr) => {
                    visit_o!(label, |label| vis.visit_label(label));
                    visit_o!(expr, |expr| vis.visit_expr(expr));
                }
                ExprKind::Continue(label) => {
                    visit_o!(label, |label| vis.visit_label(label));
                }
                ExprKind::Ret(expr)
                | ExprKind::Yeet(expr)
                | ExprKind::Yield(expr) => {
                    visit_o!(expr, |expr| vis.visit_expr(expr));
                }
                ExprKind::Become(expr) => {
                    try_v!(vis.visit_expr(expr))
                }
                ExprKind::InlineAsm(asm) => {
                    try_v!(vis.visit_inline_asm(asm))
                }
                ExprKind::FormatArgs(fmt) => {
                    try_v!(vis.visit_format_args(fmt))
                }
                ExprKind::OffsetOf(container, fields) => {
                    try_v!(vis.visit_ty(container));
                    let fields = macro_if!{$($mut)? {
                        fields.iter_mut()
                    } else {
                        fields.iter()
                    }};
                    visit_list!(vis, visit_ident, fields);
                }
                ExprKind::MacCall(mac) => {
                    try_v!(vis.visit_mac_call(mac))
                }
                ExprKind::Struct(se) => {
                    let StructExpr { qself, path, fields, rest } = &$($mut)? **se;
                    try_v!(vis.visit_qself(qself));
                    try_v!(vis.visit_path(path, *id));
                    visit_list!(vis, visit_expr_field, flat_map_expr_field, fields);
                    match rest {
                        StructRest::Base(expr) => {
                            try_v!(vis.visit_expr(expr));
                        }
                        StructRest::Rest(span) => {
                            try_v!(visit_span!(vis, span));
                        }
                        StructRest::None => {}
                    }
                }
                ExprKind::Paren(expr) => {
                    try_v!(vis.visit_expr(expr));
                }
                ExprKind::Try(expr) => {
                    try_v!(vis.visit_expr(expr));
                }
                ExprKind::TryBlock(expr) => {
                    try_v!(vis.visit_block(expr));
                }
                ExprKind::Lit(_token) => {}
                ExprKind::IncludedBytes(_bytes) => {}
                ExprKind::Err(_guar) => {}
                ExprKind::Dummy => {}
            }
            visit_lazy_tts!(vis, tokens);
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_pat<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            pattern: ref_t!(Pat)
        ) -> result!(V) {
            let Pat { id, kind, span, tokens } = pattern;
            try_v!(visit_id!(vis, id));
            match kind {
                PatKind::Err(_guar) => {}
                PatKind::Wild | PatKind::Rest | PatKind::Never => {}
                PatKind::Ident(_binding_mode, ident, sub) => {
                    try_v!(vis.visit_ident(ident));
                    visit_o!(sub, |sub| vis.visit_pat(sub));
                }
                PatKind::Lit(e) => {
                    try_v!(vis.visit_expr(e));
                }
                PatKind::TupleStruct(qself, path, elems) => {
                    try_v!(vis.visit_qself(qself));
                    try_v!(vis.visit_path(path, *id));
                    visit_list!(vis, visit_pat, elems);
                }
                PatKind::Path(qself, path) => {
                    try_v!(vis.visit_qself(qself));
                    try_v!(vis.visit_path(path, *id));
                }
                PatKind::Struct(qself, path, fields, _etc) => {
                    try_v!(vis.visit_qself(qself));
                    try_v!(vis.visit_path(path, *id));
                    visit_list!(vis, visit_pat_field, flat_map_pat_field, fields);
                }
                PatKind::Box(inner) | PatKind::Deref(inner) | PatKind::Paren(inner) => {
                    try_v!(vis.visit_pat(inner));
                }
                PatKind::Ref(inner, _mutbl) => {
                    try_v!(vis.visit_pat(inner));
                }
                PatKind::Range(e1, e2, Spanned { span, node: _ }) => {
                    visit_o!(e1, |e| vis.visit_expr(e));
                    visit_o!(e2, |e| vis.visit_expr(e));
                    try_v!(visit_span!(vis, span));
                }
                PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
                    visit_list!(vis, visit_pat, elems);
                }
                PatKind::MacCall(mac) => {
                    try_v!(vis.visit_mac_call(mac));
                }
            }
            visit_lazy_tts!(vis, tokens);
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub fn walk_ty<$($lt,)? V: $trait$(<$lt>)?>(
            vis: &mut V,
            ty: ref_t!(Ty)
        ) -> result!(V) {
            let Ty { id, kind, span, tokens } = ty;
            try_v!(visit_id!(vis, id));
            match kind {
                TyKind::Err(_guar) => {}
                TyKind::Infer
                | TyKind::ImplicitSelf
                | TyKind::Dummy
                | TyKind::Never
                | TyKind::CVarArgs => {}
                TyKind::Slice(ty) => {
                    try_v!(vis.visit_ty(ty));
                }
                TyKind::Ptr(mt) => {
                    try_v!(vis.visit_mt(mt));
                }
                TyKind::Ref(lt, mt) | TyKind::PinnedRef(lt, mt) => {
                    visit_o!(lt, |lt| vis.visit_lifetime(lt, LifetimeCtxt::Ref));
                    try_v!(vis.visit_mt(mt));
                }
                TyKind::BareFn(bft) => {
                    let BareFnTy { safety, ext: _, generic_params, decl, decl_span } = & $($mut)? **bft;
                    try_v!(vis.visit_safety(safety));
                    visit_list!(vis, visit_generic_param, flat_map_generic_param, generic_params);
                    try_v!(vis.visit_fn_decl(decl));
                    try_v!(visit_span!(vis, decl_span));
                }
                TyKind::Tup(tys) => {
                    visit_list!(vis, visit_ty, tys);
                }
                TyKind::Paren(ty) => {
                    try_v!(vis.visit_ty(ty))
                }
                TyKind::Pat(ty, pat) => {
                    try_v!(vis.visit_ty(ty));
                    try_v!(vis.visit_pat(pat));
                }
                TyKind::Path(qself, path) => {
                    try_v!(vis.visit_qself(qself));
                    try_v!(vis.visit_path(path, *id));
                }
                TyKind::Array(ty, length) => {
                    try_v!(vis.visit_ty(ty));
                    try_v!(vis.visit_anon_const(length));
                }
                TyKind::Typeof(expr) => {
                    try_v!(vis.visit_anon_const(expr));
                },
                TyKind::TraitObject(bounds, _syntax) => {
                    visit_list!(vis, visit_param_bound, bounds; BoundKind::TraitObject);
                }
                TyKind::ImplTrait(id, bounds) => {
                    try_v!(visit_id!(vis, id));
                    visit_list!(vis, visit_param_bound, bounds; BoundKind::Impl);
                }
                TyKind::MacCall(mac) => {
                    try_v!(vis.visit_mac_call(mac))
                }
            }
            visit_lazy_tts!(vis, tokens);
            try_v!(visit_span!(vis, span));
            return_result!(V)
        }

        pub trait WalkItemKind: Sized {
            fn walk<$($lt,)? V: $trait$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                id: NodeId,
                span: Span,
                vis: ref_t!(Visibility),
                ident: ref_t!(Ident),
                visitor: &mut V,
            ) -> result!(V);
        }

        impl WalkItemKind for ItemKind {
            fn walk<$($lt,)? V: $trait$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                id: NodeId,
                span: Span,
                vis: ref_t!(Visibility),
                ident: ref_t!(Ident),
                visitor: &mut V,
            ) -> result!(V) {
                match self {
                    ItemKind::ExternCrate(_orig_name) => {}
                    ItemKind::Use(use_tree) => {
                        try_v!(visitor.visit_use_tree(use_tree, id, false));
                    }
                    ItemKind::Static(box StaticItem { safety, mutability: _, ty, expr }) => {
                        try_v!(visitor.visit_safety(safety));
                        try_v!(visitor.visit_ty(ty));
                        visit_o!(expr, |expr| visitor.visit_expr(expr));
                    }
                    ItemKind::Const(box ConstItem { defaultness, generics, ty, expr }) => {
                        try_v!(visitor.visit_defaultness(defaultness));
                        try_v!(visitor.visit_generics(generics));
                        try_v!(visitor.visit_ty(ty));
                        visit_o!(expr, |expr| visitor.visit_expr(expr));
                    }
                    ItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
                        try_v!(visitor.visit_defaultness(defaultness));
                        let kind = FnKind::Fn(FnCtxt::Free, ident, sig, vis, generics, body);
                        try_v!(visitor.visit_fn(kind, span, id));
                    }
                    ItemKind::Mod(safety, mod_kind) => {
                        try_v!(visitor.visit_safety(safety));
                        match mod_kind {
                            ModKind::Loaded(
                                items,
                                _inline,
                                ModSpans { inner_span, inject_use_span },
                            ) => {
                                visit_list!(visitor, visit_item, flat_map_item, items);
                                try_v!(visit_span!(visitor, inner_span));
                                try_v!(visit_span!(visitor, inject_use_span));
                            }
                            ModKind::Unloaded => {}
                        }
                    }
                    ItemKind::ForeignMod(foreign_mod) => {
                        try_v!(visitor.visit_foreign_mod(foreign_mod));
                    }
                    ItemKind::GlobalAsm(asm) => {
                        try_v!(visitor.visit_inline_asm(asm))
                    }
                    ItemKind::TyAlias(box TyAlias {
                        defaultness,
                        generics,
                        bounds,
                        ty,
                        where_clauses,
                    }) => {
                        try_v!(visitor.visit_defaultness(defaultness));
                        try_v!(visitor.visit_generics(generics));
                        visit_list!(visitor, visit_param_bound, bounds; BoundKind::Bound);
                        visit_o!(ty, |ty| visitor.visit_ty(ty));
                        try_v!(visitor.visit_ty_alias_where_clauses(where_clauses));
                    }
                    ItemKind::Enum(enum_definition, generics) => {
                        try_v!(visitor.visit_generics(generics));
                        try_v!(visitor.visit_enum_def(enum_definition));
                    }
                    ItemKind::Struct(struct_definition, generics)
                    | ItemKind::Union(struct_definition, generics) => {
                        try_v!(visitor.visit_generics(generics));
                        try_v!(visitor.visit_variant_data(struct_definition));
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
                        try_v!(visitor.visit_defaultness(defaultness));
                        try_v!(visitor.visit_safety(safety));
                        try_v!(visitor.visit_generics(generics));
                        try_v!(visitor.visit_constness(constness));
                        try_v!(visitor.visit_impl_polarity(polarity));
                        visit_o!(of_trait, |trait_ref| visitor.visit_trait_ref(trait_ref));
                        try_v!(visitor.visit_ty(self_ty));
                        visit_list!(visitor, visit_assoc_item, flat_map_assoc_item, items; AssocCtxt::Impl);
                    }
                    ItemKind::Trait(box Trait { safety, is_auto: _, generics, bounds, items }) => {
                        try_v!(visitor.visit_safety(safety));
                        try_v!(visitor.visit_generics(generics));
                        visit_list!(visitor, visit_param_bound, bounds; BoundKind::Bound);
                        visit_list!(visitor, visit_assoc_item, flat_map_assoc_item, items; AssocCtxt::Trait);
                    }
                    ItemKind::TraitAlias(generics, bounds) => {
                        try_v!(visitor.visit_generics(generics));
                        visit_list!(visitor, visit_param_bound, bounds; BoundKind::Bound);
                    }
                    ItemKind::MacCall(mac) => {
                        try_v!(visitor.visit_mac_call(mac))
                    }
                    ItemKind::MacroDef(ts) => {
                        try_v!(visitor.visit_macro_def(ts, id));
                    }
                    ItemKind::Delegation(box Delegation {
                        id,
                        qself,
                        path,
                        rename,
                        body,
                        from_glob: _,
                    }) => {
                        try_v!(visit_id!(visitor, id));
                        try_v!(visitor.visit_qself(qself));
                        try_v!(visitor.visit_path(path, *id));
                        visit_o!(rename, |rename| visitor.visit_ident(rename));
                        visit_o!(body, |body| visitor.visit_block(body));
                    }
                    ItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                        try_v!(visitor.visit_qself(qself));
                        try_v!(visitor.visit_path(prefix, id));
                        if let Some(suffixes) = suffixes {
                            for (ident, rename) in suffixes {
                                try_v!(visitor.visit_ident(ident));
                                visit_o!(rename, |rename| visitor.visit_ident(rename));
                            }
                        }
                        visit_o!(body, |body| visitor.visit_block(body));
                    }
                }
                return_result!(V)
            }
        }

        impl WalkItemKind for ForeignItemKind {
            fn walk<$($lt,)? V: $trait$(<$lt>)?>(
                &$($lt)? $($mut)? self,
                id: NodeId,
                span: Span,
                vis: ref_t!(Visibility),
                ident: ref_t!(Ident),
                visitor: &mut V,
            ) -> result!(V) {
                match self {
                    ForeignItemKind::Static(box StaticItem { safety, ty, mutability: _, expr }) => {
                        try_v!(visitor.visit_safety(safety));
                        try_v!(visitor.visit_ty(ty));
                        visit_o!(expr, |expr| visitor.visit_expr(expr));
                    }
                    ForeignItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
                        try_v!(visitor.visit_defaultness(defaultness));
                        let kind =
                            FnKind::Fn(FnCtxt::Foreign, ident, sig, vis, generics, body);
                        visitor.visit_fn(kind, span, id);
                    }
                    ForeignItemKind::TyAlias(box TyAlias {
                        defaultness,
                        generics,
                        where_clauses,
                        bounds,
                        ty,
                    }) => {
                        try_v!(visitor.visit_defaultness(defaultness));
                        try_v!(visitor.visit_generics(generics));
                        visit_list!(visitor, visit_param_bound, bounds; BoundKind::Bound);
                        visit_o!(ty, |ty| visitor.visit_ty(ty));
                        try_v!(visitor.visit_ty_alias_where_clauses(where_clauses));
                    }
                    ForeignItemKind::MacCall(mac) => {
                        try_v!(visitor.visit_mac_call(mac));
                    }
                }
                return_result!(V)
            }
        }

        pub fn walk_item<$($lt,)? V: $trait$(<$lt>)?>(
            visitor: &mut V,
            item: ref_t!(Item<impl WalkItemKind>),
        ) -> result!(V) {
            let Item { id, span, ident, vis, attrs, kind, tokens } = item;
            try_v!(visit_id!(visitor, id));
            visit_list!(visitor, visit_attribute, attrs);
            try_v!(visitor.visit_vis(vis));
            try_v!(visitor.visit_ident(ident));
            try_v!(kind.walk(*id, *span, vis, ident, visitor));
            visit_lazy_tts!(visitor, tokens);
            try_v!(visit_span!(visitor, span));
            return_result!(V)
        }

        pub fn walk_assoc_item<$($lt,)? V: $trait$(<$lt>)?>(
            visitor: &mut V,
            item: ref_t!(Item<AssocItemKind>),
            ctxt: AssocCtxt
        ) -> result!(V) {
            let Item { attrs, id, span, vis, ident, kind, tokens } = item;
            try_v!(visit_id!(visitor, id));
            visit_list!(visitor, visit_attribute, attrs);
            try_v!(visitor.visit_vis(vis));
            try_v!(visitor.visit_ident(ident));
            match kind {
                AssocItemKind::Const(box ConstItem { defaultness, generics, ty, expr }) => {
                    try_v!(visitor.visit_defaultness(defaultness));
                    try_v!(visitor.visit_generics(generics));
                    try_v!(visitor.visit_ty(ty));
                    visit_o!(expr, |expr| visitor.visit_expr(expr));
                }
                AssocItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
                    try_v!(visitor.visit_defaultness(defaultness));
                    let kind =
                        FnKind::Fn(FnCtxt::Assoc(ctxt), ident, sig, vis, generics, body);
                    try_v!(visitor.visit_fn(kind, *span, *id));
                }
                AssocItemKind::Type(box TyAlias {
                    defaultness,
                    generics,
                    where_clauses,
                    bounds,
                    ty,
                }) => {
                    try_v!(visitor.visit_defaultness(defaultness));
                    try_v!(visitor.visit_generics(generics));
                    visit_list!(visitor, visit_param_bound, bounds; BoundKind::Bound);
                    visit_o!(ty, |ty| visitor.visit_ty(ty));
                    try_v!(visitor.visit_ty_alias_where_clauses(where_clauses));
                }
                AssocItemKind::MacCall(mac) => {
                    try_v!(visitor.visit_mac_call(mac));
                }
                AssocItemKind::Delegation(box Delegation {
                    id,
                    qself,
                    path,
                    rename,
                    body,
                    from_glob: _,
                }) => {
                    try_v!(visit_id!(visitor, id));
                    try_v!(visitor.visit_qself(qself));
                    try_v!(visitor.visit_path(path, *id));
                    visit_o!(rename, |rename| visitor.visit_ident(rename));
                    visit_o!(body, |body| visitor.visit_block(body));
                }
                AssocItemKind::DelegationMac(box DelegationMac {
                    qself,
                    prefix,
                    suffixes,
                    body,
                }) => {
                    try_v!(visitor.visit_qself(qself));
                    try_v!(visitor.visit_path(prefix, *id));
                    if let Some(suffixes) = suffixes {
                        for (ident, rename) in suffixes {
                            try_v!(visitor.visit_ident(ident));
                            visit_o!(rename, |rename| visitor.visit_ident(rename));
                        }
                    }
                    visit_o!(body, |body| visitor.visit_block(body));
                }
            }
            visit_lazy_tts!(visitor, tokens);
            try_v!(visit_span!(visitor, span));
            return_result!(V)
        }

        pub fn walk_fn<$($lt,)? V: $trait$(<$lt>)?>(
            visitor: &mut V,
            kind: FnKind!()
        ) -> result!(V) {
            match kind {
                FnKind::Fn(_ctxt, _ident, FnSig { header, decl, span }, _vis, generics, body) => {
                    // Identifier and visibility are visited as a part of the item.
                    try_v!(visitor.visit_fn_header(header));
                    try_v!(visitor.visit_generics(generics));
                    try_v!(visitor.visit_fn_decl(decl));
                    visit_o!(body, |body| visitor.visit_block(body));
                    try_v!(visit_span!(visitor, span))
                }
                FnKind::Closure(binder, coroutine_kind, decl, body) => {
                    visit_o!(coroutine_kind, |ck| visitor.visit_coroutine_kind(ck));
                    try_v!(visitor.visit_closure_binder(binder));
                    try_v!(visitor.visit_fn_decl(decl));
                    try_v!(visitor.visit_expr(body));
                }
            }
            return_result!(V)
        }

        macro_if!{$($mut)? {
            macro_rules! make_walk_flat_map {
                (
                    $ty: ty
                    $$(, $arg: ident : $arg_ty: ty)*;
                    $walk_flat_map: ident,
                    $visit: ident
                ) => {
                    pub fn $walk_flat_map(
                        vis: &mut impl $trait$(<$lt>)?,
                        mut arg: $ty
                        $$(, $arg: $arg_ty)*
                    ) -> SmallVec<[$ty; 1]> {
                        vis.$visit(&mut arg $$(, $arg)*);
                        smallvec![arg]
                    }
                }
            }

            make_walk_flat_map!{Arm; walk_flat_map_arm, visit_arm}
            make_walk_flat_map!{ExprField; walk_flat_map_expr_field, visit_expr_field}
            make_walk_flat_map!{FieldDef; walk_flat_map_field_def, visit_field_def}
            make_walk_flat_map!{GenericParam; walk_flat_map_generic_param, visit_generic_param}
            make_walk_flat_map!{Param; walk_flat_map_param, visit_param}
            make_walk_flat_map!{PatField; walk_flat_map_pat_field, visit_pat_field}
            make_walk_flat_map!{Variant; walk_flat_map_variant, visit_variant}

            make_walk_flat_map!{P<Item>; walk_flat_map_item, visit_item}
            make_walk_flat_map!{P<AssocItem>, ctxt: AssocCtxt; walk_flat_map_assoc_item, visit_assoc_item}
            make_walk_flat_map!{P<ForeignItem>; walk_flat_map_foreign_item, visit_foreign_item}
        }}
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

    make_ast_visitor!(Visitor<'ast>);

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
    use crate::visit::{AssocCtxt, BoundKind, FnCtxt, LifetimeCtxt};

    pub trait ExpectOne<A: Array> {
        fn expect_one(self, err: &'static str) -> A::Item;
    }

    impl<A: Array> ExpectOne<A> for SmallVec<A> {
        fn expect_one(self, err: &'static str) -> A::Item {
            assert!(self.len() == 1, "{}", err);
            self.into_iter().next().unwrap()
        }
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
    fn visit_delim_args<T: MutVisitor>(vis: &mut T, args: &mut DelimArgs) {
        let DelimArgs { dspan, delim: _, tokens } = args;
        visit_tts(vis, tokens);
        visit_delim_span(vis, dspan);
    }

    pub fn visit_delim_span<T: MutVisitor>(vis: &mut T, DelimSpan { open, close }: &mut DelimSpan) {
        vis.visit_span(open);
        vis.visit_span(close);
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
                vis.visit_path(path, DUMMY_NODE_ID);
                vis.visit_attr_args(args);
                visit_lazy_tts(vis, tokens);
            }
            token::NtPath(path) => vis.visit_path(path, DUMMY_NODE_ID),
            token::NtVis(visib) => vis.visit_vis(visib),
        }
    }

    pub fn walk_item_kind(item: &mut Item<impl WalkItemKind>, vis: &mut impl MutVisitor) {
        item.kind.walk(item.id, item.span, &mut item.vis, &mut item.ident, vis)
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
}
