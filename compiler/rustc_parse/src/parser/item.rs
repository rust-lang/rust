use std::fmt::Write;
use std::mem;

use ast::token::IdentIsRaw;
use rustc_ast::ast::*;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, InvisibleOrigin, MetaVarKind, TokenKind};
use rustc_ast::tokenstream::{DelimSpan, TokenStream, TokenTree};
use rustc_ast::util::case::Case;
use rustc_ast::{self as ast};
use rustc_ast_pretty::pprust;
use rustc_errors::codes::*;
use rustc_errors::{Applicability, PResult, StashKey, struct_span_code_err};
use rustc_span::edit_distance::edit_distance;
use rustc_span::edition::Edition;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Ident, Span, Symbol, kw, source_map, sym};
use thin_vec::{ThinVec, thin_vec};
use tracing::debug;

use super::diagnostics::{ConsumeClosingDelim, dummy_arg};
use super::ty::{AllowPlus, RecoverQPath, RecoverReturnSign};
use super::{
    AttrWrapper, ExpKeywordPair, ExpTokenPair, FollowedByType, ForceCollect, Parser, PathStyle,
    Recovered, Trailing, UsePreAttrPos,
};
use crate::errors::{self, MacroExpandsToAdtField};
use crate::{exp, fluent_generated as fluent};

impl<'a> Parser<'a> {
    /// Parses a source module as a crate. This is the main entry point for the parser.
    pub fn parse_crate_mod(&mut self) -> PResult<'a, ast::Crate> {
        let (attrs, items, spans) = self.parse_mod(exp!(Eof))?;
        Ok(ast::Crate { attrs, items, spans, id: DUMMY_NODE_ID, is_placeholder: false })
    }

    /// Parses a `mod <foo> { ... }` or `mod <foo>;` item.
    fn parse_item_mod(&mut self, attrs: &mut AttrVec) -> PResult<'a, ItemKind> {
        let safety = self.parse_safety(Case::Sensitive);
        self.expect_keyword(exp!(Mod))?;
        let ident = self.parse_ident()?;
        let mod_kind = if self.eat(exp!(Semi)) {
            ModKind::Unloaded
        } else {
            self.expect(exp!(OpenBrace))?;
            let (inner_attrs, items, inner_span) = self.parse_mod(exp!(CloseBrace))?;
            attrs.extend(inner_attrs);
            ModKind::Loaded(items, Inline::Yes, inner_span, Ok(()))
        };
        Ok(ItemKind::Mod(safety, ident, mod_kind))
    }

    /// Parses the contents of a module (inner attributes followed by module items).
    /// We exit once we hit `term` which can be either
    /// - EOF (for files)
    /// - `}` for mod items
    pub fn parse_mod(
        &mut self,
        term: ExpTokenPair<'_>,
    ) -> PResult<'a, (AttrVec, ThinVec<P<Item>>, ModSpans)> {
        let lo = self.token.span;
        let attrs = self.parse_inner_attributes()?;

        let post_attr_lo = self.token.span;
        let mut items: ThinVec<P<_>> = ThinVec::new();

        // There shouldn't be any stray semicolons before or after items.
        // `parse_item` consumes the appropriate semicolons so any leftover is an error.
        loop {
            while self.maybe_consume_incorrect_semicolon(items.last().map(|x| &**x)) {} // Eat all bad semicolons
            let Some(item) = self.parse_item(ForceCollect::No)? else {
                break;
            };
            items.push(item);
        }

        if !self.eat(term) {
            let token_str = super::token_descr(&self.token);
            if !self.maybe_consume_incorrect_semicolon(items.last().map(|x| &**x)) {
                let is_let = self.token.is_keyword(kw::Let);
                let is_let_mut = is_let && self.look_ahead(1, |t| t.is_keyword(kw::Mut));
                let let_has_ident = is_let && !is_let_mut && self.is_kw_followed_by_ident(kw::Let);

                let msg = format!("expected item, found {token_str}");
                let mut err = self.dcx().struct_span_err(self.token.span, msg);

                let label = if is_let {
                    "`let` cannot be used for global variables"
                } else {
                    "expected item"
                };
                err.span_label(self.token.span, label);

                if is_let {
                    if is_let_mut {
                        err.help("consider using `static` and a `Mutex` instead of `let mut`");
                    } else if let_has_ident {
                        err.span_suggestion_short(
                            self.token.span,
                            "consider using `static` or `const` instead of `let`",
                            "static",
                            Applicability::MaybeIncorrect,
                        );
                    } else {
                        err.help("consider using `static` or `const` instead of `let`");
                    }
                }
                err.note("for a full list of items that can appear in modules, see <https://doc.rust-lang.org/reference/items.html>");
                return Err(err);
            }
        }

        let inject_use_span = post_attr_lo.data().with_hi(post_attr_lo.lo());
        let mod_spans = ModSpans { inner_span: lo.to(self.prev_token.span), inject_use_span };
        Ok((attrs, items, mod_spans))
    }
}

impl<'a> Parser<'a> {
    pub fn parse_item(&mut self, force_collect: ForceCollect) -> PResult<'a, Option<P<Item>>> {
        let fn_parse_mode = FnParseMode { req_name: |_| true, req_body: true };
        self.parse_item_(fn_parse_mode, force_collect).map(|i| i.map(P))
    }

    fn parse_item_(
        &mut self,
        fn_parse_mode: FnParseMode,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Item>> {
        self.recover_vcs_conflict_marker();
        let attrs = self.parse_outer_attributes()?;
        self.recover_vcs_conflict_marker();
        self.parse_item_common(attrs, true, false, fn_parse_mode, force_collect)
    }

    pub(super) fn parse_item_common(
        &mut self,
        attrs: AttrWrapper,
        mac_allowed: bool,
        attrs_allowed: bool,
        fn_parse_mode: FnParseMode,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Item>> {
        if let Some(item) =
            self.eat_metavar_seq(MetaVarKind::Item, |this| this.parse_item(ForceCollect::Yes))
        {
            let mut item = item.expect("an actual item");
            attrs.prepend_to_nt_inner(&mut item.attrs);
            return Ok(Some(item.into_inner()));
        }

        self.collect_tokens(None, attrs, force_collect, |this, mut attrs| {
            let lo = this.token.span;
            let vis = this.parse_visibility(FollowedByType::No)?;
            let mut def = this.parse_defaultness();
            let kind = this.parse_item_kind(
                &mut attrs,
                mac_allowed,
                lo,
                &vis,
                &mut def,
                fn_parse_mode,
                Case::Sensitive,
            )?;
            if let Some(kind) = kind {
                this.error_on_unconsumed_default(def, &kind);
                let span = lo.to(this.prev_token.span);
                let id = DUMMY_NODE_ID;
                let item = Item { attrs, id, kind, vis, span, tokens: None };
                return Ok((Some(item), Trailing::No, UsePreAttrPos::No));
            }

            // At this point, we have failed to parse an item.
            if !matches!(vis.kind, VisibilityKind::Inherited) {
                this.dcx().emit_err(errors::VisibilityNotFollowedByItem { span: vis.span, vis });
            }

            if let Defaultness::Default(span) = def {
                this.dcx().emit_err(errors::DefaultNotFollowedByItem { span });
            }

            if !attrs_allowed {
                this.recover_attrs_no_item(&attrs)?;
            }
            Ok((None, Trailing::No, UsePreAttrPos::No))
        })
    }

    /// Error in-case `default` was parsed in an in-appropriate context.
    fn error_on_unconsumed_default(&self, def: Defaultness, kind: &ItemKind) {
        if let Defaultness::Default(span) = def {
            self.dcx().emit_err(errors::InappropriateDefault {
                span,
                article: kind.article(),
                descr: kind.descr(),
            });
        }
    }

    /// Parses one of the items allowed by the flags.
    fn parse_item_kind(
        &mut self,
        attrs: &mut AttrVec,
        macros_allowed: bool,
        lo: Span,
        vis: &Visibility,
        def: &mut Defaultness,
        fn_parse_mode: FnParseMode,
        case: Case,
    ) -> PResult<'a, Option<ItemKind>> {
        let check_pub = def == &Defaultness::Final;
        let mut def_ = || mem::replace(def, Defaultness::Final);

        let info = if !self.is_use_closure() && self.eat_keyword_case(exp!(Use), case) {
            self.parse_use_item()?
        } else if self.check_fn_front_matter(check_pub, case) {
            // FUNCTION ITEM
            let (ident, sig, generics, contract, body) =
                self.parse_fn(attrs, fn_parse_mode, lo, vis, case)?;
            ItemKind::Fn(Box::new(Fn {
                defaultness: def_(),
                ident,
                sig,
                generics,
                contract,
                body,
                define_opaque: None,
            }))
        } else if self.eat_keyword(exp!(Extern)) {
            if self.eat_keyword(exp!(Crate)) {
                // EXTERN CRATE
                self.parse_item_extern_crate()?
            } else {
                // EXTERN BLOCK
                self.parse_item_foreign_mod(attrs, Safety::Default)?
            }
        } else if self.is_unsafe_foreign_mod() {
            // EXTERN BLOCK
            let safety = self.parse_safety(Case::Sensitive);
            self.expect_keyword(exp!(Extern))?;
            self.parse_item_foreign_mod(attrs, safety)?
        } else if self.is_static_global() {
            let safety = self.parse_safety(Case::Sensitive);
            // STATIC ITEM
            self.bump(); // `static`
            let mutability = self.parse_mutability();
            self.parse_static_item(safety, mutability)?
        } else if let Const::Yes(const_span) = self.parse_constness(Case::Sensitive) {
            // CONST ITEM
            if self.token.is_keyword(kw::Impl) {
                // recover from `const impl`, suggest `impl const`
                self.recover_const_impl(const_span, attrs, def_())?
            } else {
                self.recover_const_mut(const_span);
                self.recover_missing_kw_before_item()?;
                let (ident, generics, ty, expr) = self.parse_const_item()?;
                ItemKind::Const(Box::new(ConstItem {
                    defaultness: def_(),
                    ident,
                    generics,
                    ty,
                    expr,
                    define_opaque: None,
                }))
            }
        } else if self.check_keyword(exp!(Trait)) || self.check_auto_or_unsafe_trait_item() {
            // TRAIT ITEM
            self.parse_item_trait(attrs, lo)?
        } else if self.check_keyword(exp!(Impl))
            || self.check_keyword(exp!(Unsafe)) && self.is_keyword_ahead(1, &[kw::Impl])
        {
            // IMPL ITEM
            self.parse_item_impl(attrs, def_())?
        } else if self.is_reuse_path_item() {
            self.parse_item_delegation()?
        } else if self.check_keyword(exp!(Mod))
            || self.check_keyword(exp!(Unsafe)) && self.is_keyword_ahead(1, &[kw::Mod])
        {
            // MODULE ITEM
            self.parse_item_mod(attrs)?
        } else if self.eat_keyword(exp!(Type)) {
            // TYPE ITEM
            self.parse_type_alias(def_())?
        } else if self.eat_keyword(exp!(Enum)) {
            // ENUM ITEM
            self.parse_item_enum()?
        } else if self.eat_keyword(exp!(Struct)) {
            // STRUCT ITEM
            self.parse_item_struct()?
        } else if self.is_kw_followed_by_ident(kw::Union) {
            // UNION ITEM
            self.bump(); // `union`
            self.parse_item_union()?
        } else if self.is_builtin() {
            // BUILTIN# ITEM
            return self.parse_item_builtin();
        } else if self.eat_keyword(exp!(Macro)) {
            // MACROS 2.0 ITEM
            self.parse_item_decl_macro(lo)?
        } else if let IsMacroRulesItem::Yes { has_bang } = self.is_macro_rules_item() {
            // MACRO_RULES ITEM
            self.parse_item_macro_rules(vis, has_bang)?
        } else if self.isnt_macro_invocation()
            && (self.token.is_ident_named(sym::import)
                || self.token.is_ident_named(sym::using)
                || self.token.is_ident_named(sym::include)
                || self.token.is_ident_named(sym::require))
        {
            return self.recover_import_as_use();
        } else if self.isnt_macro_invocation() && vis.kind.is_pub() {
            self.recover_missing_kw_before_item()?;
            return Ok(None);
        } else if self.isnt_macro_invocation() && case == Case::Sensitive {
            _ = def_;

            // Recover wrong cased keywords
            return self.parse_item_kind(
                attrs,
                macros_allowed,
                lo,
                vis,
                def,
                fn_parse_mode,
                Case::Insensitive,
            );
        } else if macros_allowed && self.check_path() {
            if self.isnt_macro_invocation() {
                self.recover_missing_kw_before_item()?;
            }
            // MACRO INVOCATION ITEM
            ItemKind::MacCall(P(self.parse_item_macro(vis)?))
        } else {
            return Ok(None);
        };
        Ok(Some(info))
    }

    fn recover_import_as_use(&mut self) -> PResult<'a, Option<ItemKind>> {
        let span = self.token.span;
        let token_name = super::token_descr(&self.token);
        let snapshot = self.create_snapshot_for_diagnostic();
        self.bump();
        match self.parse_use_item() {
            Ok(u) => {
                self.dcx().emit_err(errors::RecoverImportAsUse { span, token_name });
                Ok(Some(u))
            }
            Err(e) => {
                e.cancel();
                self.restore_snapshot(snapshot);
                Ok(None)
            }
        }
    }

    fn parse_use_item(&mut self) -> PResult<'a, ItemKind> {
        let tree = self.parse_use_tree()?;
        if let Err(mut e) = self.expect_semi() {
            match tree.kind {
                UseTreeKind::Glob => {
                    e.note("the wildcard token must be last on the path");
                }
                UseTreeKind::Nested { .. } => {
                    e.note("glob-like brace syntax must be last on the path");
                }
                _ => (),
            }
            return Err(e);
        }
        Ok(ItemKind::Use(tree))
    }

    /// When parsing a statement, would the start of a path be an item?
    pub(super) fn is_path_start_item(&mut self) -> bool {
        self.is_kw_followed_by_ident(kw::Union) // no: `union::b`, yes: `union U { .. }`
        || self.is_reuse_path_item()
        || self.check_auto_or_unsafe_trait_item() // no: `auto::b`, yes: `auto trait X { .. }`
        || self.is_async_fn() // no(2015): `async::b`, yes: `async fn`
        || matches!(self.is_macro_rules_item(), IsMacroRulesItem::Yes{..}) // no: `macro_rules::b`, yes: `macro_rules! mac`
    }

    fn is_reuse_path_item(&mut self) -> bool {
        // no: `reuse ::path` for compatibility reasons with macro invocations
        self.token.is_keyword(kw::Reuse)
            && self.look_ahead(1, |t| t.is_path_start() && *t != token::PathSep)
    }

    /// Are we sure this could not possibly be a macro invocation?
    fn isnt_macro_invocation(&mut self) -> bool {
        self.check_ident() && self.look_ahead(1, |t| *t != token::Bang && *t != token::PathSep)
    }

    /// Recover on encountering a struct, enum, or method definition where the user
    /// forgot to add the `struct`, `enum`, or `fn` keyword
    fn recover_missing_kw_before_item(&mut self) -> PResult<'a, ()> {
        let is_pub = self.prev_token.is_keyword(kw::Pub);
        let is_const = self.prev_token.is_keyword(kw::Const);
        let ident_span = self.token.span;
        let span = if is_pub { self.prev_token.span.to(ident_span) } else { ident_span };
        let insert_span = ident_span.shrink_to_lo();

        let ident = if self.token.is_ident()
            && (!is_const || self.look_ahead(1, |t| *t == token::OpenDelim(Delimiter::Parenthesis)))
            && self.look_ahead(1, |t| {
                [
                    token::Lt,
                    token::OpenDelim(Delimiter::Brace),
                    token::OpenDelim(Delimiter::Parenthesis),
                ]
                .contains(&t.kind)
            }) {
            self.parse_ident().unwrap()
        } else {
            return Ok(());
        };

        let mut found_generics = false;
        if self.check(exp!(Lt)) {
            found_generics = true;
            self.eat_to_tokens(&[exp!(Gt)]);
            self.bump(); // `>`
        }

        let err = if self.check(exp!(OpenBrace)) {
            // possible struct or enum definition where `struct` or `enum` was forgotten
            if self.look_ahead(1, |t| *t == token::CloseDelim(Delimiter::Brace)) {
                // `S {}` could be unit enum or struct
                Some(errors::MissingKeywordForItemDefinition::EnumOrStruct { span })
            } else if self.look_ahead(2, |t| *t == token::Colon)
                || self.look_ahead(3, |t| *t == token::Colon)
            {
                // `S { f:` or `S { pub f:`
                Some(errors::MissingKeywordForItemDefinition::Struct { span, insert_span, ident })
            } else {
                Some(errors::MissingKeywordForItemDefinition::Enum { span, insert_span, ident })
            }
        } else if self.check(exp!(OpenParen)) {
            // possible function or tuple struct definition where `fn` or `struct` was forgotten
            self.bump(); // `(`
            let is_method = self.recover_self_param();

            self.consume_block(exp!(OpenParen), exp!(CloseParen), ConsumeClosingDelim::Yes);

            let err = if self.check(exp!(RArrow)) || self.check(exp!(OpenBrace)) {
                self.eat_to_tokens(&[exp!(OpenBrace)]);
                self.bump(); // `{`
                self.consume_block(exp!(OpenBrace), exp!(CloseBrace), ConsumeClosingDelim::Yes);
                if is_method {
                    errors::MissingKeywordForItemDefinition::Method { span, insert_span, ident }
                } else {
                    errors::MissingKeywordForItemDefinition::Function { span, insert_span, ident }
                }
            } else if is_pub && self.check(exp!(Semi)) {
                errors::MissingKeywordForItemDefinition::Struct { span, insert_span, ident }
            } else {
                errors::MissingKeywordForItemDefinition::Ambiguous {
                    span,
                    subdiag: if found_generics {
                        None
                    } else if let Ok(snippet) = self.span_to_snippet(ident_span) {
                        Some(errors::AmbiguousMissingKwForItemSub::SuggestMacro {
                            span: ident_span,
                            snippet,
                        })
                    } else {
                        Some(errors::AmbiguousMissingKwForItemSub::HelpMacro)
                    },
                }
            };
            Some(err)
        } else if found_generics {
            Some(errors::MissingKeywordForItemDefinition::Ambiguous { span, subdiag: None })
        } else {
            None
        };

        if let Some(err) = err { Err(self.dcx().create_err(err)) } else { Ok(()) }
    }

    fn parse_item_builtin(&mut self) -> PResult<'a, Option<ItemKind>> {
        // To be expanded
        Ok(None)
    }

    /// Parses an item macro, e.g., `item!();`.
    fn parse_item_macro(&mut self, vis: &Visibility) -> PResult<'a, MacCall> {
        let path = self.parse_path(PathStyle::Mod)?; // `foo::bar`
        self.expect(exp!(Bang))?; // `!`
        match self.parse_delim_args() {
            // `( .. )` or `[ .. ]` (followed by `;`), or `{ .. }`.
            Ok(args) => {
                self.eat_semi_for_macro_if_needed(&args);
                self.complain_if_pub_macro(vis, false);
                Ok(MacCall { path, args })
            }

            Err(mut err) => {
                // Maybe the user misspelled `macro_rules` (issue #91227)
                if self.token.is_ident()
                    && let [segment] = path.segments.as_slice()
                    && edit_distance("macro_rules", &segment.ident.to_string(), 2).is_some()
                {
                    err.span_suggestion(
                        path.span,
                        "perhaps you meant to define a macro",
                        "macro_rules",
                        Applicability::MachineApplicable,
                    );
                }
                Err(err)
            }
        }
    }

    /// Recover if we parsed attributes and expected an item but there was none.
    fn recover_attrs_no_item(&mut self, attrs: &[Attribute]) -> PResult<'a, ()> {
        let ([start @ end] | [start, .., end]) = attrs else {
            return Ok(());
        };
        let msg = if end.is_doc_comment() {
            "expected item after doc comment"
        } else {
            "expected item after attributes"
        };
        let mut err = self.dcx().struct_span_err(end.span, msg);
        if end.is_doc_comment() {
            err.span_label(end.span, "this doc comment doesn't document anything");
        } else if self.token == TokenKind::Semi {
            err.span_suggestion_verbose(
                self.token.span,
                "consider removing this semicolon",
                "",
                Applicability::MaybeIncorrect,
            );
        }
        if let [.., penultimate, _] = attrs {
            err.span_label(start.span.to(penultimate.span), "other attributes here");
        }
        Err(err)
    }

    fn is_async_fn(&self) -> bool {
        self.token.is_keyword(kw::Async) && self.is_keyword_ahead(1, &[kw::Fn])
    }

    fn parse_polarity(&mut self) -> ast::ImplPolarity {
        // Disambiguate `impl !Trait for Type { ... }` and `impl ! { ... }` for the never type.
        if self.check(exp!(Bang)) && self.look_ahead(1, |t| t.can_begin_type()) {
            self.bump(); // `!`
            ast::ImplPolarity::Negative(self.prev_token.span)
        } else {
            ast::ImplPolarity::Positive
        }
    }

    /// Parses an implementation item.
    ///
    /// ```ignore (illustrative)
    /// impl<'a, T> TYPE { /* impl items */ }
    /// impl<'a, T> TRAIT for TYPE { /* impl items */ }
    /// impl<'a, T> !TRAIT for TYPE { /* impl items */ }
    /// impl<'a, T> const TRAIT for TYPE { /* impl items */ }
    /// ```
    ///
    /// We actually parse slightly more relaxed grammar for better error reporting and recovery.
    /// ```ebnf
    /// "impl" GENERICS "const"? "!"? TYPE "for"? (TYPE | "..") ("where" PREDICATES)? "{" BODY "}"
    /// "impl" GENERICS "const"? "!"? TYPE ("where" PREDICATES)? "{" BODY "}"
    /// ```
    fn parse_item_impl(
        &mut self,
        attrs: &mut AttrVec,
        defaultness: Defaultness,
    ) -> PResult<'a, ItemKind> {
        let safety = self.parse_safety(Case::Sensitive);
        self.expect_keyword(exp!(Impl))?;

        // First, parse generic parameters if necessary.
        let mut generics = if self.choose_generics_over_qpath(0) {
            self.parse_generics()?
        } else {
            let mut generics = Generics::default();
            // impl A for B {}
            //    /\ this is where `generics.span` should point when there are no type params.
            generics.span = self.prev_token.span.shrink_to_hi();
            generics
        };

        let constness = self.parse_constness(Case::Sensitive);
        if let Const::Yes(span) = constness {
            self.psess.gated_spans.gate(sym::const_trait_impl, span);
        }

        // Parse stray `impl async Trait`
        if (self.token_uninterpolated_span().at_least_rust_2018()
            && self.token.is_keyword(kw::Async))
            || self.is_kw_followed_by_ident(kw::Async)
        {
            self.bump();
            self.dcx().emit_err(errors::AsyncImpl { span: self.prev_token.span });
        }

        let polarity = self.parse_polarity();

        // Parse both types and traits as a type, then reinterpret if necessary.
        let ty_first = if self.token.is_keyword(kw::For) && self.look_ahead(1, |t| t != &token::Lt)
        {
            let span = self.prev_token.span.between(self.token.span);
            return Err(self.dcx().create_err(errors::MissingTraitInTraitImpl {
                span,
                for_span: span.to(self.token.span),
            }));
        } else {
            self.parse_ty_with_generics_recovery(&generics)?
        };

        // If `for` is missing we try to recover.
        let has_for = self.eat_keyword(exp!(For));
        let missing_for_span = self.prev_token.span.between(self.token.span);

        let ty_second = if self.token == token::DotDot {
            // We need to report this error after `cfg` expansion for compatibility reasons
            self.bump(); // `..`, do not add it to expected tokens

            // AST validation later detects this `TyKind::Dummy` and emits an
            // error. (#121072 will hopefully remove all this special handling
            // of the obsolete `impl Trait for ..` and then this can go away.)
            Some(self.mk_ty(self.prev_token.span, TyKind::Dummy))
        } else if has_for || self.token.can_begin_type() {
            Some(self.parse_ty()?)
        } else {
            None
        };

        generics.where_clause = self.parse_where_clause()?;

        let impl_items = self.parse_item_list(attrs, |p| p.parse_impl_item(ForceCollect::No))?;

        let (of_trait, self_ty) = match ty_second {
            Some(ty_second) => {
                // impl Trait for Type
                if !has_for {
                    self.dcx().emit_err(errors::MissingForInTraitImpl { span: missing_for_span });
                }

                let ty_first = ty_first.into_inner();
                let path = match ty_first.kind {
                    // This notably includes paths passed through `ty` macro fragments (#46438).
                    TyKind::Path(None, path) => path,
                    other => {
                        if let TyKind::ImplTrait(_, bounds) = other
                            && let [bound] = bounds.as_slice()
                            && let GenericBound::Trait(poly_trait_ref) = bound
                        {
                            // Suggest removing extra `impl` keyword:
                            // `impl<T: Default> impl Default for Wrapper<T>`
                            //                   ^^^^^
                            let extra_impl_kw = ty_first.span.until(bound.span());
                            self.dcx().emit_err(errors::ExtraImplKeywordInTraitImpl {
                                extra_impl_kw,
                                impl_trait_span: ty_first.span,
                            });
                            poly_trait_ref.trait_ref.path.clone()
                        } else {
                            return Err(self.dcx().create_err(
                                errors::ExpectedTraitInTraitImplFoundType { span: ty_first.span },
                            ));
                        }
                    }
                };
                let trait_ref = TraitRef { path, ref_id: ty_first.id };

                (Some(trait_ref), ty_second)
            }
            None => (None, ty_first), // impl Type
        };
        Ok(ItemKind::Impl(Box::new(Impl {
            safety,
            polarity,
            defaultness,
            constness,
            generics,
            of_trait,
            self_ty,
            items: impl_items,
        })))
    }

    fn parse_item_delegation(&mut self) -> PResult<'a, ItemKind> {
        let span = self.token.span;
        self.expect_keyword(exp!(Reuse))?;

        let (qself, path) = if self.eat_lt() {
            let (qself, path) = self.parse_qpath(PathStyle::Expr)?;
            (Some(qself), path)
        } else {
            (None, self.parse_path(PathStyle::Expr)?)
        };

        let rename = |this: &mut Self| {
            Ok(if this.eat_keyword(exp!(As)) { Some(this.parse_ident()?) } else { None })
        };
        let body = |this: &mut Self| {
            Ok(if this.check(exp!(OpenBrace)) {
                Some(this.parse_block()?)
            } else {
                this.expect(exp!(Semi))?;
                None
            })
        };

        let item_kind = if self.eat_path_sep() {
            let suffixes = if self.eat(exp!(Star)) {
                None
            } else {
                let parse_suffix = |p: &mut Self| Ok((p.parse_path_segment_ident()?, rename(p)?));
                Some(self.parse_delim_comma_seq(exp!(OpenBrace), exp!(CloseBrace), parse_suffix)?.0)
            };
            let deleg = DelegationMac { qself, prefix: path, suffixes, body: body(self)? };
            ItemKind::DelegationMac(Box::new(deleg))
        } else {
            let rename = rename(self)?;
            let ident = rename.unwrap_or_else(|| path.segments.last().unwrap().ident);
            let deleg = Delegation {
                id: DUMMY_NODE_ID,
                qself,
                path,
                ident,
                rename,
                body: body(self)?,
                from_glob: false,
            };
            ItemKind::Delegation(Box::new(deleg))
        };

        let span = span.to(self.prev_token.span);
        self.psess.gated_spans.gate(sym::fn_delegation, span);

        Ok(item_kind)
    }

    fn parse_item_list<T>(
        &mut self,
        attrs: &mut AttrVec,
        mut parse_item: impl FnMut(&mut Parser<'a>) -> PResult<'a, Option<Option<T>>>,
    ) -> PResult<'a, ThinVec<T>> {
        let open_brace_span = self.token.span;

        // Recover `impl Ty;` instead of `impl Ty {}`
        if self.token == TokenKind::Semi {
            self.dcx().emit_err(errors::UseEmptyBlockNotSemi { span: self.token.span });
            self.bump();
            return Ok(ThinVec::new());
        }

        self.expect(exp!(OpenBrace))?;
        attrs.extend(self.parse_inner_attributes()?);

        let mut items = ThinVec::new();
        while !self.eat(exp!(CloseBrace)) {
            if self.recover_doc_comment_before_brace() {
                continue;
            }
            self.recover_vcs_conflict_marker();
            match parse_item(self) {
                Ok(None) => {
                    let mut is_unnecessary_semicolon = !items.is_empty()
                        // When the close delim is `)` in a case like the following, `token.kind` is expected to be `token::CloseDelim(Delimiter::Parenthesis)`,
                        // but the actual `token.kind` is `token::CloseDelim(Delimiter::Brace)`.
                        // This is because the `token.kind` of the close delim is treated as the same as
                        // that of the open delim in `TokenTreesReader::parse_token_tree`, even if the delimiters of them are different.
                        // Therefore, `token.kind` should not be compared here.
                        //
                        // issue-60075.rs
                        // ```
                        // trait T {
                        //     fn qux() -> Option<usize> {
                        //         let _ = if true {
                        //         });
                        //          ^ this close delim
                        //         Some(4)
                        //     }
                        // ```
                        && self
                            .span_to_snippet(self.prev_token.span)
                            .is_ok_and(|snippet| snippet == "}")
                        && self.token == token::Semi;
                    let mut semicolon_span = self.token.span;
                    if !is_unnecessary_semicolon {
                        // #105369, Detect spurious `;` before assoc fn body
                        is_unnecessary_semicolon = self.token == token::OpenDelim(Delimiter::Brace)
                            && self.prev_token == token::Semi;
                        semicolon_span = self.prev_token.span;
                    }
                    // We have to bail or we'll potentially never make progress.
                    let non_item_span = self.token.span;
                    let is_let = self.token.is_keyword(kw::Let);

                    let mut err =
                        self.dcx().struct_span_err(non_item_span, "non-item in item list");
                    self.consume_block(exp!(OpenBrace), exp!(CloseBrace), ConsumeClosingDelim::Yes);
                    if is_let {
                        err.span_suggestion_verbose(
                            non_item_span,
                            "consider using `const` instead of `let` for associated const",
                            "const",
                            Applicability::MachineApplicable,
                        );
                    } else {
                        err.span_label(open_brace_span, "item list starts here")
                            .span_label(non_item_span, "non-item starts here")
                            .span_label(self.prev_token.span, "item list ends here");
                    }
                    if is_unnecessary_semicolon {
                        err.span_suggestion(
                            semicolon_span,
                            "consider removing this semicolon",
                            "",
                            Applicability::MaybeIncorrect,
                        );
                    }
                    err.emit();
                    break;
                }
                Ok(Some(item)) => items.extend(item),
                Err(err) => {
                    self.consume_block(exp!(OpenBrace), exp!(CloseBrace), ConsumeClosingDelim::Yes);
                    err.with_span_label(
                        open_brace_span,
                        "while parsing this item list starting here",
                    )
                    .with_span_label(self.prev_token.span, "the item list ends here")
                    .emit();
                    break;
                }
            }
        }
        Ok(items)
    }

    /// Recover on a doc comment before `}`.
    fn recover_doc_comment_before_brace(&mut self) -> bool {
        if let token::DocComment(..) = self.token.kind {
            if self.look_ahead(1, |tok| tok == &token::CloseDelim(Delimiter::Brace)) {
                // FIXME: merge with `DocCommentDoesNotDocumentAnything` (E0585)
                struct_span_code_err!(
                    self.dcx(),
                    self.token.span,
                    E0584,
                    "found a documentation comment that doesn't document anything",
                )
                .with_span_label(self.token.span, "this doc comment doesn't document anything")
                .with_help(
                    "doc comments must come before what they document, if a comment was \
                    intended use `//`",
                )
                .emit();
                self.bump();
                return true;
            }
        }
        false
    }

    /// Parses defaultness (i.e., `default` or nothing).
    fn parse_defaultness(&mut self) -> Defaultness {
        // We are interested in `default` followed by another identifier.
        // However, we must avoid keywords that occur as binary operators.
        // Currently, the only applicable keyword is `as` (`default as Ty`).
        if self.check_keyword(exp!(Default))
            && self.look_ahead(1, |t| t.is_non_raw_ident_where(|i| i.name != kw::As))
        {
            self.bump(); // `default`
            Defaultness::Default(self.prev_token_uninterpolated_span())
        } else {
            Defaultness::Final
        }
    }

    /// Is this an `(unsafe auto? | auto) trait` item?
    fn check_auto_or_unsafe_trait_item(&mut self) -> bool {
        // auto trait
        self.check_keyword(exp!(Auto)) && self.is_keyword_ahead(1, &[kw::Trait])
            // unsafe auto trait
            || self.check_keyword(exp!(Unsafe)) && self.is_keyword_ahead(1, &[kw::Trait, kw::Auto])
    }

    /// Parses `unsafe? auto? trait Foo { ... }` or `trait Foo = Bar;`.
    fn parse_item_trait(&mut self, attrs: &mut AttrVec, lo: Span) -> PResult<'a, ItemKind> {
        let safety = self.parse_safety(Case::Sensitive);
        // Parse optional `auto` prefix.
        let is_auto = if self.eat_keyword(exp!(Auto)) {
            self.psess.gated_spans.gate(sym::auto_traits, self.prev_token.span);
            IsAuto::Yes
        } else {
            IsAuto::No
        };

        self.expect_keyword(exp!(Trait))?;
        let ident = self.parse_ident()?;
        let mut generics = self.parse_generics()?;

        // Parse optional colon and supertrait bounds.
        let had_colon = self.eat(exp!(Colon));
        let span_at_colon = self.prev_token.span;
        let bounds = if had_colon { self.parse_generic_bounds()? } else { Vec::new() };

        let span_before_eq = self.prev_token.span;
        if self.eat(exp!(Eq)) {
            // It's a trait alias.
            if had_colon {
                let span = span_at_colon.to(span_before_eq);
                self.dcx().emit_err(errors::BoundsNotAllowedOnTraitAliases { span });
            }

            let bounds = self.parse_generic_bounds()?;
            generics.where_clause = self.parse_where_clause()?;
            self.expect_semi()?;

            let whole_span = lo.to(self.prev_token.span);
            if is_auto == IsAuto::Yes {
                self.dcx().emit_err(errors::TraitAliasCannotBeAuto { span: whole_span });
            }
            if let Safety::Unsafe(_) = safety {
                self.dcx().emit_err(errors::TraitAliasCannotBeUnsafe { span: whole_span });
            }

            self.psess.gated_spans.gate(sym::trait_alias, whole_span);

            Ok(ItemKind::TraitAlias(ident, generics, bounds))
        } else {
            // It's a normal trait.
            generics.where_clause = self.parse_where_clause()?;
            let items = self.parse_item_list(attrs, |p| p.parse_trait_item(ForceCollect::No))?;
            Ok(ItemKind::Trait(Box::new(Trait { is_auto, safety, ident, generics, bounds, items })))
        }
    }

    pub fn parse_impl_item(
        &mut self,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Option<P<AssocItem>>>> {
        let fn_parse_mode = FnParseMode { req_name: |_| true, req_body: true };
        self.parse_assoc_item(fn_parse_mode, force_collect)
    }

    pub fn parse_trait_item(
        &mut self,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Option<P<AssocItem>>>> {
        let fn_parse_mode =
            FnParseMode { req_name: |edition| edition >= Edition::Edition2018, req_body: false };
        self.parse_assoc_item(fn_parse_mode, force_collect)
    }

    /// Parses associated items.
    fn parse_assoc_item(
        &mut self,
        fn_parse_mode: FnParseMode,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Option<P<AssocItem>>>> {
        Ok(self.parse_item_(fn_parse_mode, force_collect)?.map(
            |Item { attrs, id, span, vis, kind, tokens }| {
                let kind = match AssocItemKind::try_from(kind) {
                    Ok(kind) => kind,
                    Err(kind) => match kind {
                        ItemKind::Static(box StaticItem {
                            ident,
                            ty,
                            safety: _,
                            mutability: _,
                            expr,
                            define_opaque,
                        }) => {
                            self.dcx().emit_err(errors::AssociatedStaticItemNotAllowed { span });
                            AssocItemKind::Const(Box::new(ConstItem {
                                defaultness: Defaultness::Final,
                                ident,
                                generics: Generics::default(),
                                ty,
                                expr,
                                define_opaque,
                            }))
                        }
                        _ => return self.error_bad_item_kind(span, &kind, "`trait`s or `impl`s"),
                    },
                };
                Some(P(Item { attrs, id, span, vis, kind, tokens }))
            },
        ))
    }

    /// Parses a `type` alias with the following grammar:
    /// ```ebnf
    /// TypeAlias = "type" Ident Generics (":" GenericBounds)? WhereClause ("=" Ty)? WhereClause ";" ;
    /// ```
    /// The `"type"` has already been eaten.
    fn parse_type_alias(&mut self, defaultness: Defaultness) -> PResult<'a, ItemKind> {
        let ident = self.parse_ident()?;
        let mut generics = self.parse_generics()?;

        // Parse optional colon and param bounds.
        let bounds = if self.eat(exp!(Colon)) { self.parse_generic_bounds()? } else { Vec::new() };
        let before_where_clause = self.parse_where_clause()?;

        let ty = if self.eat(exp!(Eq)) { Some(self.parse_ty()?) } else { None };

        let after_where_clause = self.parse_where_clause()?;

        let where_clauses = TyAliasWhereClauses {
            before: TyAliasWhereClause {
                has_where_token: before_where_clause.has_where_token,
                span: before_where_clause.span,
            },
            after: TyAliasWhereClause {
                has_where_token: after_where_clause.has_where_token,
                span: after_where_clause.span,
            },
            split: before_where_clause.predicates.len(),
        };
        let mut predicates = before_where_clause.predicates;
        predicates.extend(after_where_clause.predicates);
        let where_clause = WhereClause {
            has_where_token: before_where_clause.has_where_token
                || after_where_clause.has_where_token,
            predicates,
            span: DUMMY_SP,
        };
        generics.where_clause = where_clause;

        self.expect_semi()?;

        Ok(ItemKind::TyAlias(Box::new(TyAlias {
            defaultness,
            ident,
            generics,
            where_clauses,
            bounds,
            ty,
        })))
    }

    /// Parses a `UseTree`.
    ///
    /// ```text
    /// USE_TREE = [`::`] `*` |
    ///            [`::`] `{` USE_TREE_LIST `}` |
    ///            PATH `::` `*` |
    ///            PATH `::` `{` USE_TREE_LIST `}` |
    ///            PATH [`as` IDENT]
    /// ```
    fn parse_use_tree(&mut self) -> PResult<'a, UseTree> {
        let lo = self.token.span;

        let mut prefix =
            ast::Path { segments: ThinVec::new(), span: lo.shrink_to_lo(), tokens: None };
        let kind =
            if self.check(exp!(OpenBrace)) || self.check(exp!(Star)) || self.is_import_coupler() {
                // `use *;` or `use ::*;` or `use {...};` or `use ::{...};`
                let mod_sep_ctxt = self.token.span.ctxt();
                if self.eat_path_sep() {
                    prefix
                        .segments
                        .push(PathSegment::path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt)));
                }

                self.parse_use_tree_glob_or_nested()?
            } else {
                // `use path::*;` or `use path::{...};` or `use path;` or `use path as bar;`
                prefix = self.parse_path(PathStyle::Mod)?;

                if self.eat_path_sep() {
                    self.parse_use_tree_glob_or_nested()?
                } else {
                    // Recover from using a colon as path separator.
                    while self.eat_noexpect(&token::Colon) {
                        self.dcx()
                            .emit_err(errors::SingleColonImportPath { span: self.prev_token.span });

                        // We parse the rest of the path and append it to the original prefix.
                        self.parse_path_segments(&mut prefix.segments, PathStyle::Mod, None)?;
                        prefix.span = lo.to(self.prev_token.span);
                    }

                    UseTreeKind::Simple(self.parse_rename()?)
                }
            };

        Ok(UseTree { prefix, kind, span: lo.to(self.prev_token.span) })
    }

    /// Parses `*` or `{...}`.
    fn parse_use_tree_glob_or_nested(&mut self) -> PResult<'a, UseTreeKind> {
        Ok(if self.eat(exp!(Star)) {
            UseTreeKind::Glob
        } else {
            let lo = self.token.span;
            UseTreeKind::Nested {
                items: self.parse_use_tree_list()?,
                span: lo.to(self.prev_token.span),
            }
        })
    }

    /// Parses a `UseTreeKind::Nested(list)`.
    ///
    /// ```text
    /// USE_TREE_LIST = ∅ | (USE_TREE `,`)* USE_TREE [`,`]
    /// ```
    fn parse_use_tree_list(&mut self) -> PResult<'a, ThinVec<(UseTree, ast::NodeId)>> {
        self.parse_delim_comma_seq(exp!(OpenBrace), exp!(CloseBrace), |p| {
            p.recover_vcs_conflict_marker();
            Ok((p.parse_use_tree()?, DUMMY_NODE_ID))
        })
        .map(|(r, _)| r)
    }

    fn parse_rename(&mut self) -> PResult<'a, Option<Ident>> {
        if self.eat_keyword(exp!(As)) {
            self.parse_ident_or_underscore().map(Some)
        } else {
            Ok(None)
        }
    }

    fn parse_ident_or_underscore(&mut self) -> PResult<'a, Ident> {
        match self.token.ident() {
            Some((ident @ Ident { name: kw::Underscore, .. }, IdentIsRaw::No)) => {
                self.bump();
                Ok(ident)
            }
            _ => self.parse_ident(),
        }
    }

    /// Parses `extern crate` links.
    ///
    /// # Examples
    ///
    /// ```ignore (illustrative)
    /// extern crate foo;
    /// extern crate bar as foo;
    /// ```
    fn parse_item_extern_crate(&mut self) -> PResult<'a, ItemKind> {
        // Accept `extern crate name-like-this` for better diagnostics
        let orig_ident = self.parse_crate_name_with_dashes()?;
        let (orig_name, item_ident) = if let Some(rename) = self.parse_rename()? {
            (Some(orig_ident.name), rename)
        } else {
            (None, orig_ident)
        };
        self.expect_semi()?;
        Ok(ItemKind::ExternCrate(orig_name, item_ident))
    }

    fn parse_crate_name_with_dashes(&mut self) -> PResult<'a, Ident> {
        let ident = if self.token.is_keyword(kw::SelfLower) {
            self.parse_path_segment_ident()
        } else {
            self.parse_ident()
        }?;

        let dash = exp!(Minus);
        if self.token != *dash.tok {
            return Ok(ident);
        }

        // Accept `extern crate name-like-this` for better diagnostics.
        let mut dashes = vec![];
        let mut idents = vec![];
        while self.eat(dash) {
            dashes.push(self.prev_token.span);
            idents.push(self.parse_ident()?);
        }

        let fixed_name_sp = ident.span.to(idents.last().unwrap().span);
        let mut fixed_name = ident.name.to_string();
        for part in idents {
            write!(fixed_name, "_{}", part.name).unwrap();
        }

        self.dcx().emit_err(errors::ExternCrateNameWithDashes {
            span: fixed_name_sp,
            sugg: errors::ExternCrateNameWithDashesSugg { dashes },
        });

        Ok(Ident::from_str_and_span(&fixed_name, fixed_name_sp))
    }

    /// Parses `extern` for foreign ABIs modules.
    ///
    /// `extern` is expected to have been consumed before calling this method.
    ///
    /// # Examples
    ///
    /// ```ignore (only-for-syntax-highlight)
    /// extern "C" {}
    /// extern {}
    /// ```
    fn parse_item_foreign_mod(
        &mut self,
        attrs: &mut AttrVec,
        mut safety: Safety,
    ) -> PResult<'a, ItemKind> {
        let extern_span = self.prev_token_uninterpolated_span();
        let abi = self.parse_abi(); // ABI?
        // FIXME: This recovery should be tested better.
        if safety == Safety::Default
            && self.token.is_keyword(kw::Unsafe)
            && self.look_ahead(1, |t| *t == token::OpenDelim(Delimiter::Brace))
        {
            self.expect(exp!(OpenBrace)).unwrap_err().emit();
            safety = Safety::Unsafe(self.token.span);
            let _ = self.eat_keyword(exp!(Unsafe));
        }
        Ok(ItemKind::ForeignMod(ast::ForeignMod {
            extern_span,
            safety,
            abi,
            items: self.parse_item_list(attrs, |p| p.parse_foreign_item(ForceCollect::No))?,
        }))
    }

    /// Parses a foreign item (one in an `extern { ... }` block).
    pub fn parse_foreign_item(
        &mut self,
        force_collect: ForceCollect,
    ) -> PResult<'a, Option<Option<P<ForeignItem>>>> {
        let fn_parse_mode = FnParseMode { req_name: |_| true, req_body: false };
        Ok(self.parse_item_(fn_parse_mode, force_collect)?.map(
            |Item { attrs, id, span, vis, kind, tokens }| {
                let kind = match ForeignItemKind::try_from(kind) {
                    Ok(kind) => kind,
                    Err(kind) => match kind {
                        ItemKind::Const(box ConstItem { ident, ty, expr, .. }) => {
                            let const_span = Some(span.with_hi(ident.span.lo()))
                                .filter(|span| span.can_be_used_for_suggestions());
                            self.dcx().emit_err(errors::ExternItemCannotBeConst {
                                ident_span: ident.span,
                                const_span,
                            });
                            ForeignItemKind::Static(Box::new(StaticItem {
                                ident,
                                ty,
                                mutability: Mutability::Not,
                                expr,
                                safety: Safety::Default,
                                define_opaque: None,
                            }))
                        }
                        _ => return self.error_bad_item_kind(span, &kind, "`extern` blocks"),
                    },
                };
                Some(P(Item { attrs, id, span, vis, kind, tokens }))
            },
        ))
    }

    fn error_bad_item_kind<T>(&self, span: Span, kind: &ItemKind, ctx: &'static str) -> Option<T> {
        // FIXME(#100717): needs variant for each `ItemKind` (instead of using `ItemKind::descr()`)
        let span = self.psess.source_map().guess_head_span(span);
        let descr = kind.descr();
        let help = match kind {
            ItemKind::DelegationMac(deleg) if deleg.suffixes.is_none() => false,
            _ => true,
        };
        self.dcx().emit_err(errors::BadItemKind { span, descr, ctx, help });
        None
    }

    fn is_use_closure(&self) -> bool {
        if self.token.is_keyword(kw::Use) {
            // Check if this could be a closure.
            self.look_ahead(1, |token| {
                // Move or Async here would be an error but still we're parsing a closure
                let dist =
                    if token.is_keyword(kw::Move) || token.is_keyword(kw::Async) { 2 } else { 1 };

                self.look_ahead(dist, |token| matches!(token.kind, token::Or | token::OrOr))
            })
        } else {
            false
        }
    }

    fn is_unsafe_foreign_mod(&self) -> bool {
        // Look for `unsafe`.
        if !self.token.is_keyword(kw::Unsafe) {
            return false;
        }
        // Look for `extern`.
        if !self.is_keyword_ahead(1, &[kw::Extern]) {
            return false;
        }

        // Look for the optional ABI string literal.
        let n = if self.look_ahead(2, |t| t.can_begin_string_literal()) { 3 } else { 2 };

        // Look for the `{`. Use `tree_look_ahead` because the ABI (if present)
        // might be a metavariable i.e. an invisible-delimited sequence, and
        // `tree_look_ahead` will consider that a single element when looking
        // ahead.
        self.tree_look_ahead(n, |t| matches!(t, TokenTree::Delimited(_, _, Delimiter::Brace, _)))
            == Some(true)
    }

    fn is_static_global(&mut self) -> bool {
        if self.check_keyword(exp!(Static)) {
            // Check if this could be a closure.
            !self.look_ahead(1, |token| {
                if token.is_keyword(kw::Move) || token.is_keyword(kw::Use) {
                    return true;
                }
                matches!(token.kind, token::Or | token::OrOr)
            })
        } else {
            // `$qual static`
            (self.check_keyword(exp!(Unsafe)) || self.check_keyword(exp!(Safe)))
                && self.look_ahead(1, |t| t.is_keyword(kw::Static))
        }
    }

    /// Recover on `const mut` with `const` already eaten.
    fn recover_const_mut(&mut self, const_span: Span) {
        if self.eat_keyword(exp!(Mut)) {
            let span = self.prev_token.span;
            self.dcx()
                .emit_err(errors::ConstGlobalCannotBeMutable { ident_span: span, const_span });
        } else if self.eat_keyword(exp!(Let)) {
            let span = self.prev_token.span;
            self.dcx().emit_err(errors::ConstLetMutuallyExclusive { span: const_span.to(span) });
        }
    }

    /// Recover on `const impl` with `const` already eaten.
    fn recover_const_impl(
        &mut self,
        const_span: Span,
        attrs: &mut AttrVec,
        defaultness: Defaultness,
    ) -> PResult<'a, ItemKind> {
        let impl_span = self.token.span;
        let err = self.expected_ident_found_err();

        // Only try to recover if this is implementing a trait for a type
        let mut item_kind = match self.parse_item_impl(attrs, defaultness) {
            Ok(item_kind) => item_kind,
            Err(recovery_error) => {
                // Recovery failed, raise the "expected identifier" error
                recovery_error.cancel();
                return Err(err);
            }
        };

        match &mut item_kind {
            ItemKind::Impl(box Impl { of_trait: Some(trai), constness, .. }) => {
                *constness = Const::Yes(const_span);

                let before_trait = trai.path.span.shrink_to_lo();
                let const_up_to_impl = const_span.with_hi(impl_span.lo());
                err.with_multipart_suggestion(
                    "you might have meant to write a const trait impl",
                    vec![(const_up_to_impl, "".to_owned()), (before_trait, "const ".to_owned())],
                    Applicability::MaybeIncorrect,
                )
                .emit();
            }
            ItemKind::Impl { .. } => return Err(err),
            _ => unreachable!(),
        }

        Ok(item_kind)
    }

    /// Parse a static item with the prefix `"static" "mut"?` already parsed and stored in
    /// `mutability`.
    ///
    /// ```ebnf
    /// Static = "static" "mut"? $ident ":" $ty (= $expr)? ";" ;
    /// ```
    fn parse_static_item(
        &mut self,
        safety: Safety,
        mutability: Mutability,
    ) -> PResult<'a, ItemKind> {
        let ident = self.parse_ident()?;

        if self.token == TokenKind::Lt && self.may_recover() {
            let generics = self.parse_generics()?;
            self.dcx().emit_err(errors::StaticWithGenerics { span: generics.span });
        }

        // Parse the type of a static item. That is, the `":" $ty` fragment.
        // FIXME: This could maybe benefit from `.may_recover()`?
        let ty = match (self.eat(exp!(Colon)), self.check(exp!(Eq)) | self.check(exp!(Semi))) {
            (true, false) => self.parse_ty()?,
            // If there wasn't a `:` or the colon was followed by a `=` or `;`, recover a missing
            // type.
            (colon, _) => self.recover_missing_global_item_type(colon, Some(mutability)),
        };

        let expr = if self.eat(exp!(Eq)) { Some(self.parse_expr()?) } else { None };

        self.expect_semi()?;

        let item = StaticItem { ident, ty, safety, mutability, expr, define_opaque: None };
        Ok(ItemKind::Static(Box::new(item)))
    }

    /// Parse a constant item with the prefix `"const"` already parsed.
    ///
    /// ```ebnf
    /// Const = "const" ($ident | "_") Generics ":" $ty (= $expr)? WhereClause ";" ;
    /// ```
    fn parse_const_item(&mut self) -> PResult<'a, (Ident, Generics, P<Ty>, Option<P<ast::Expr>>)> {
        let ident = self.parse_ident_or_underscore()?;

        let mut generics = self.parse_generics()?;

        // Check the span for emptiness instead of the list of parameters in order to correctly
        // recognize and subsequently flag empty parameter lists (`<>`) as unstable.
        if !generics.span.is_empty() {
            self.psess.gated_spans.gate(sym::generic_const_items, generics.span);
        }

        // Parse the type of a constant item. That is, the `":" $ty` fragment.
        // FIXME: This could maybe benefit from `.may_recover()`?
        let ty = match (
            self.eat(exp!(Colon)),
            self.check(exp!(Eq)) | self.check(exp!(Semi)) | self.check_keyword(exp!(Where)),
        ) {
            (true, false) => self.parse_ty()?,
            // If there wasn't a `:` or the colon was followed by a `=`, `;` or `where`, recover a missing type.
            (colon, _) => self.recover_missing_global_item_type(colon, None),
        };

        // Proactively parse a where-clause to be able to provide a good error message in case we
        // encounter the item body following it.
        let before_where_clause =
            if self.may_recover() { self.parse_where_clause()? } else { WhereClause::default() };

        let expr = if self.eat(exp!(Eq)) { Some(self.parse_expr()?) } else { None };

        let after_where_clause = self.parse_where_clause()?;

        // Provide a nice error message if the user placed a where-clause before the item body.
        // Users may be tempted to write such code if they are still used to the deprecated
        // where-clause location on type aliases and associated types. See also #89122.
        if before_where_clause.has_where_token
            && let Some(expr) = &expr
        {
            self.dcx().emit_err(errors::WhereClauseBeforeConstBody {
                span: before_where_clause.span,
                name: ident.span,
                body: expr.span,
                sugg: if !after_where_clause.has_where_token {
                    self.psess.source_map().span_to_snippet(expr.span).ok().map(|body| {
                        errors::WhereClauseBeforeConstBodySugg {
                            left: before_where_clause.span.shrink_to_lo(),
                            snippet: body,
                            right: before_where_clause.span.shrink_to_hi().to(expr.span),
                        }
                    })
                } else {
                    // FIXME(generic_const_items): Provide a structured suggestion to merge the first
                    // where-clause into the second one.
                    None
                },
            });
        }

        // Merge the predicates of both where-clauses since either one can be relevant.
        // If we didn't parse a body (which is valid for associated consts in traits) and we were
        // allowed to recover, `before_where_clause` contains the predicates, otherwise they are
        // in `after_where_clause`. Further, both of them might contain predicates iff two
        // where-clauses were provided which is syntactically ill-formed but we want to recover from
        // it and treat them as one large where-clause.
        let mut predicates = before_where_clause.predicates;
        predicates.extend(after_where_clause.predicates);
        let where_clause = WhereClause {
            has_where_token: before_where_clause.has_where_token
                || after_where_clause.has_where_token,
            predicates,
            span: if after_where_clause.has_where_token {
                after_where_clause.span
            } else {
                before_where_clause.span
            },
        };

        if where_clause.has_where_token {
            self.psess.gated_spans.gate(sym::generic_const_items, where_clause.span);
        }

        generics.where_clause = where_clause;

        self.expect_semi()?;

        Ok((ident, generics, ty, expr))
    }

    /// We were supposed to parse `":" $ty` but the `:` or the type was missing.
    /// This means that the type is missing.
    fn recover_missing_global_item_type(
        &mut self,
        colon_present: bool,
        m: Option<Mutability>,
    ) -> P<Ty> {
        // Construct the error and stash it away with the hope
        // that typeck will later enrich the error with a type.
        let kind = match m {
            Some(Mutability::Mut) => "static mut",
            Some(Mutability::Not) => "static",
            None => "const",
        };

        let colon = match colon_present {
            true => "",
            false => ":",
        };

        let span = self.prev_token.span.shrink_to_hi();
        let err = self.dcx().create_err(errors::MissingConstType { span, colon, kind });
        err.stash(span, StashKey::ItemNoType);

        // The user intended that the type be inferred,
        // so treat this as if the user wrote e.g. `const A: _ = expr;`.
        P(Ty { kind: TyKind::Infer, span, id: ast::DUMMY_NODE_ID, tokens: None })
    }

    /// Parses an enum declaration.
    fn parse_item_enum(&mut self) -> PResult<'a, ItemKind> {
        if self.token.is_keyword(kw::Struct) {
            let span = self.prev_token.span.to(self.token.span);
            let err = errors::EnumStructMutuallyExclusive { span };
            if self.look_ahead(1, |t| t.is_ident()) {
                self.bump();
                self.dcx().emit_err(err);
            } else {
                return Err(self.dcx().create_err(err));
            }
        }

        let prev_span = self.prev_token.span;
        let ident = self.parse_ident()?;
        let mut generics = self.parse_generics()?;
        generics.where_clause = self.parse_where_clause()?;

        // Possibly recover `enum Foo;` instead of `enum Foo {}`
        let (variants, _) = if self.token == TokenKind::Semi {
            self.dcx().emit_err(errors::UseEmptyBlockNotSemi { span: self.token.span });
            self.bump();
            (thin_vec![], Trailing::No)
        } else {
            self.parse_delim_comma_seq(exp!(OpenBrace), exp!(CloseBrace), |p| {
                p.parse_enum_variant(ident.span)
            })
            .map_err(|mut err| {
                err.span_label(ident.span, "while parsing this enum");
                if self.token == token::Colon {
                    let snapshot = self.create_snapshot_for_diagnostic();
                    self.bump();
                    match self.parse_ty() {
                        Ok(_) => {
                            err.span_suggestion_verbose(
                                prev_span,
                                "perhaps you meant to use `struct` here",
                                "struct",
                                Applicability::MaybeIncorrect,
                            );
                        }
                        Err(e) => {
                            e.cancel();
                        }
                    }
                    self.restore_snapshot(snapshot);
                }
                self.eat_to_tokens(&[exp!(CloseBrace)]);
                self.bump(); // }
                err
            })?
        };

        let enum_definition = EnumDef { variants: variants.into_iter().flatten().collect() };
        Ok(ItemKind::Enum(ident, enum_definition, generics))
    }

    fn parse_enum_variant(&mut self, span: Span) -> PResult<'a, Option<Variant>> {
        self.recover_vcs_conflict_marker();
        let variant_attrs = self.parse_outer_attributes()?;
        self.recover_vcs_conflict_marker();
        let help = "enum variants can be `Variant`, `Variant = <integer>`, \
                    `Variant(Type, ..., TypeN)` or `Variant { fields: Types }`";
        self.collect_tokens(None, variant_attrs, ForceCollect::No, |this, variant_attrs| {
            let vlo = this.token.span;

            let vis = this.parse_visibility(FollowedByType::No)?;
            if !this.recover_nested_adt_item(kw::Enum)? {
                return Ok((None, Trailing::No, UsePreAttrPos::No));
            }
            let ident = this.parse_field_ident("enum", vlo)?;

            if this.token == token::Bang {
                if let Err(err) = this.unexpected() {
                    err.with_note(fluent::parse_macro_expands_to_enum_variant).emit();
                }

                this.bump();
                this.parse_delim_args()?;

                return Ok((None, Trailing::from(this.token == token::Comma), UsePreAttrPos::No));
            }

            let struct_def = if this.check(exp!(OpenBrace)) {
                // Parse a struct variant.
                let (fields, recovered) =
                    match this.parse_record_struct_body("struct", ident.span, false) {
                        Ok((fields, recovered)) => (fields, recovered),
                        Err(mut err) => {
                            if this.token == token::Colon {
                                // We handle `enum` to `struct` suggestion in the caller.
                                return Err(err);
                            }
                            this.eat_to_tokens(&[exp!(CloseBrace)]);
                            this.bump(); // }
                            err.span_label(span, "while parsing this enum");
                            err.help(help);
                            let guar = err.emit();
                            (thin_vec![], Recovered::Yes(guar))
                        }
                    };
                VariantData::Struct { fields, recovered }
            } else if this.check(exp!(OpenParen)) {
                let body = match this.parse_tuple_struct_body() {
                    Ok(body) => body,
                    Err(mut err) => {
                        if this.token == token::Colon {
                            // We handle `enum` to `struct` suggestion in the caller.
                            return Err(err);
                        }
                        this.eat_to_tokens(&[exp!(CloseParen)]);
                        this.bump(); // )
                        err.span_label(span, "while parsing this enum");
                        err.help(help);
                        err.emit();
                        thin_vec![]
                    }
                };
                VariantData::Tuple(body, DUMMY_NODE_ID)
            } else {
                VariantData::Unit(DUMMY_NODE_ID)
            };

            let disr_expr =
                if this.eat(exp!(Eq)) { Some(this.parse_expr_anon_const()?) } else { None };

            let vr = ast::Variant {
                ident,
                vis,
                id: DUMMY_NODE_ID,
                attrs: variant_attrs,
                data: struct_def,
                disr_expr,
                span: vlo.to(this.prev_token.span),
                is_placeholder: false,
            };

            Ok((Some(vr), Trailing::from(this.token == token::Comma), UsePreAttrPos::No))
        })
        .map_err(|mut err| {
            err.help(help);
            err
        })
    }

    /// Parses `struct Foo { ... }`.
    fn parse_item_struct(&mut self) -> PResult<'a, ItemKind> {
        let ident = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        // There is a special case worth noting here, as reported in issue #17904.
        // If we are parsing a tuple struct it is the case that the where clause
        // should follow the field list. Like so:
        //
        // struct Foo<T>(T) where T: Copy;
        //
        // If we are parsing a normal record-style struct it is the case
        // that the where clause comes before the body, and after the generics.
        // So if we look ahead and see a brace or a where-clause we begin
        // parsing a record style struct.
        //
        // Otherwise if we look ahead and see a paren we parse a tuple-style
        // struct.

        let vdata = if self.token.is_keyword(kw::Where) {
            let tuple_struct_body;
            (generics.where_clause, tuple_struct_body) =
                self.parse_struct_where_clause(ident, generics.span)?;

            if let Some(body) = tuple_struct_body {
                // If we see a misplaced tuple struct body: `struct Foo<T> where T: Copy, (T);`
                let body = VariantData::Tuple(body, DUMMY_NODE_ID);
                self.expect_semi()?;
                body
            } else if self.eat(exp!(Semi)) {
                // If we see a: `struct Foo<T> where T: Copy;` style decl.
                VariantData::Unit(DUMMY_NODE_ID)
            } else {
                // If we see: `struct Foo<T> where T: Copy { ... }`
                let (fields, recovered) = self.parse_record_struct_body(
                    "struct",
                    ident.span,
                    generics.where_clause.has_where_token,
                )?;
                VariantData::Struct { fields, recovered }
            }
        // No `where` so: `struct Foo<T>;`
        } else if self.eat(exp!(Semi)) {
            VariantData::Unit(DUMMY_NODE_ID)
        // Record-style struct definition
        } else if self.token == token::OpenDelim(Delimiter::Brace) {
            let (fields, recovered) = self.parse_record_struct_body(
                "struct",
                ident.span,
                generics.where_clause.has_where_token,
            )?;
            VariantData::Struct { fields, recovered }
        // Tuple-style struct definition with optional where-clause.
        } else if self.token == token::OpenDelim(Delimiter::Parenthesis) {
            let body = VariantData::Tuple(self.parse_tuple_struct_body()?, DUMMY_NODE_ID);
            generics.where_clause = self.parse_where_clause()?;
            self.expect_semi()?;
            body
        } else {
            let err = errors::UnexpectedTokenAfterStructName::new(self.token.span, self.token);
            return Err(self.dcx().create_err(err));
        };

        Ok(ItemKind::Struct(ident, vdata, generics))
    }

    /// Parses `union Foo { ... }`.
    fn parse_item_union(&mut self) -> PResult<'a, ItemKind> {
        let ident = self.parse_ident()?;

        let mut generics = self.parse_generics()?;

        let vdata = if self.token.is_keyword(kw::Where) {
            generics.where_clause = self.parse_where_clause()?;
            let (fields, recovered) = self.parse_record_struct_body(
                "union",
                ident.span,
                generics.where_clause.has_where_token,
            )?;
            VariantData::Struct { fields, recovered }
        } else if self.token == token::OpenDelim(Delimiter::Brace) {
            let (fields, recovered) = self.parse_record_struct_body(
                "union",
                ident.span,
                generics.where_clause.has_where_token,
            )?;
            VariantData::Struct { fields, recovered }
        } else {
            let token_str = super::token_descr(&self.token);
            let msg = format!("expected `where` or `{{` after union name, found {token_str}");
            let mut err = self.dcx().struct_span_err(self.token.span, msg);
            err.span_label(self.token.span, "expected `where` or `{` after union name");
            return Err(err);
        };

        Ok(ItemKind::Union(ident, vdata, generics))
    }

    /// This function parses the fields of record structs:
    ///
    ///   - `struct S { ... }`
    ///   - `enum E { Variant { ... } }`
    pub(crate) fn parse_record_struct_body(
        &mut self,
        adt_ty: &str,
        ident_span: Span,
        parsed_where: bool,
    ) -> PResult<'a, (ThinVec<FieldDef>, Recovered)> {
        let mut fields = ThinVec::new();
        let mut recovered = Recovered::No;
        if self.eat(exp!(OpenBrace)) {
            while self.token != token::CloseDelim(Delimiter::Brace) {
                match self.parse_field_def(adt_ty) {
                    Ok(field) => {
                        fields.push(field);
                    }
                    Err(mut err) => {
                        self.consume_block(
                            exp!(OpenBrace),
                            exp!(CloseBrace),
                            ConsumeClosingDelim::No,
                        );
                        err.span_label(ident_span, format!("while parsing this {adt_ty}"));
                        let guar = err.emit();
                        recovered = Recovered::Yes(guar);
                        break;
                    }
                }
            }
            self.expect(exp!(CloseBrace))?;
        } else {
            let token_str = super::token_descr(&self.token);
            let where_str = if parsed_where { "" } else { "`where`, or " };
            let msg = format!("expected {where_str}`{{` after struct name, found {token_str}");
            let mut err = self.dcx().struct_span_err(self.token.span, msg);
            err.span_label(self.token.span, format!("expected {where_str}`{{` after struct name",));
            return Err(err);
        }

        Ok((fields, recovered))
    }

    fn parse_unsafe_field(&mut self) -> Safety {
        // not using parse_safety as that also accepts `safe`.
        if self.eat_keyword(exp!(Unsafe)) {
            let span = self.prev_token.span;
            self.psess.gated_spans.gate(sym::unsafe_fields, span);
            Safety::Unsafe(span)
        } else {
            Safety::Default
        }
    }

    pub(super) fn parse_tuple_struct_body(&mut self) -> PResult<'a, ThinVec<FieldDef>> {
        // This is the case where we find `struct Foo<T>(T) where T: Copy;`
        // Unit like structs are handled in parse_item_struct function
        self.parse_paren_comma_seq(|p| {
            let attrs = p.parse_outer_attributes()?;
            p.collect_tokens(None, attrs, ForceCollect::No, |p, attrs| {
                let mut snapshot = None;
                if p.is_vcs_conflict_marker(&TokenKind::Shl, &TokenKind::Lt) {
                    // Account for `<<<<<<<` diff markers. We can't proactively error here because
                    // that can be a valid type start, so we snapshot and reparse only we've
                    // encountered another parse error.
                    snapshot = Some(p.create_snapshot_for_diagnostic());
                }
                let lo = p.token.span;
                let vis = match p.parse_visibility(FollowedByType::Yes) {
                    Ok(vis) => vis,
                    Err(err) => {
                        if let Some(ref mut snapshot) = snapshot {
                            snapshot.recover_vcs_conflict_marker();
                        }
                        return Err(err);
                    }
                };
                // Unsafe fields are not supported in tuple structs, as doing so would result in a
                // parsing ambiguity for `struct X(unsafe fn())`.
                let ty = match p.parse_ty() {
                    Ok(ty) => ty,
                    Err(err) => {
                        if let Some(ref mut snapshot) = snapshot {
                            snapshot.recover_vcs_conflict_marker();
                        }
                        return Err(err);
                    }
                };
                let mut default = None;
                if p.token == token::Eq {
                    let mut snapshot = p.create_snapshot_for_diagnostic();
                    snapshot.bump();
                    match snapshot.parse_expr_anon_const() {
                        Ok(const_expr) => {
                            let sp = ty.span.shrink_to_hi().to(const_expr.value.span);
                            p.psess.gated_spans.gate(sym::default_field_values, sp);
                            p.restore_snapshot(snapshot);
                            default = Some(const_expr);
                        }
                        Err(err) => {
                            err.cancel();
                        }
                    }
                }

                Ok((
                    FieldDef {
                        span: lo.to(ty.span),
                        vis,
                        safety: Safety::Default,
                        ident: None,
                        id: DUMMY_NODE_ID,
                        ty,
                        default,
                        attrs,
                        is_placeholder: false,
                    },
                    Trailing::from(p.token == token::Comma),
                    UsePreAttrPos::No,
                ))
            })
        })
        .map(|(r, _)| r)
    }

    /// Parses an element of a struct declaration.
    fn parse_field_def(&mut self, adt_ty: &str) -> PResult<'a, FieldDef> {
        self.recover_vcs_conflict_marker();
        let attrs = self.parse_outer_attributes()?;
        self.recover_vcs_conflict_marker();
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            let lo = this.token.span;
            let vis = this.parse_visibility(FollowedByType::No)?;
            let safety = this.parse_unsafe_field();
            this.parse_single_struct_field(adt_ty, lo, vis, safety, attrs)
                .map(|field| (field, Trailing::No, UsePreAttrPos::No))
        })
    }

    /// Parses a structure field declaration.
    fn parse_single_struct_field(
        &mut self,
        adt_ty: &str,
        lo: Span,
        vis: Visibility,
        safety: Safety,
        attrs: AttrVec,
    ) -> PResult<'a, FieldDef> {
        let mut seen_comma: bool = false;
        let a_var = self.parse_name_and_ty(adt_ty, lo, vis, safety, attrs)?;
        if self.token == token::Comma {
            seen_comma = true;
        }
        if self.eat(exp!(Semi)) {
            let sp = self.prev_token.span;
            let mut err =
                self.dcx().struct_span_err(sp, format!("{adt_ty} fields are separated by `,`"));
            err.span_suggestion_short(
                sp,
                "replace `;` with `,`",
                ",",
                Applicability::MachineApplicable,
            );
            return Err(err);
        }
        match self.token.kind {
            token::Comma => {
                self.bump();
            }
            token::CloseDelim(Delimiter::Brace) => {}
            token::DocComment(..) => {
                let previous_span = self.prev_token.span;
                let mut err = errors::DocCommentDoesNotDocumentAnything {
                    span: self.token.span,
                    missing_comma: None,
                };
                self.bump(); // consume the doc comment
                let comma_after_doc_seen = self.eat(exp!(Comma));
                // `seen_comma` is always false, because we are inside doc block
                // condition is here to make code more readable
                if !seen_comma && comma_after_doc_seen {
                    seen_comma = true;
                }
                if comma_after_doc_seen || self.token == token::CloseDelim(Delimiter::Brace) {
                    self.dcx().emit_err(err);
                } else {
                    if !seen_comma {
                        let sp = previous_span.shrink_to_hi();
                        err.missing_comma = Some(sp);
                    }
                    return Err(self.dcx().create_err(err));
                }
            }
            _ => {
                let sp = self.prev_token.span.shrink_to_hi();
                let msg =
                    format!("expected `,`, or `}}`, found {}", super::token_descr(&self.token));

                // Try to recover extra trailing angle brackets
                if let TyKind::Path(_, Path { segments, .. }) = &a_var.ty.kind {
                    if let Some(last_segment) = segments.last() {
                        let guar = self.check_trailing_angle_brackets(
                            last_segment,
                            &[exp!(Comma), exp!(CloseBrace)],
                        );
                        if let Some(_guar) = guar {
                            // Handle a case like `Vec<u8>>,` where we can continue parsing fields
                            // after the comma
                            let _ = self.eat(exp!(Comma));

                            // `check_trailing_angle_brackets` already emitted a nicer error, as
                            // proven by the presence of `_guar`. We can continue parsing.
                            return Ok(a_var);
                        }
                    }
                }

                let mut err = self.dcx().struct_span_err(sp, msg);

                if self.token.is_ident()
                    || (self.token == TokenKind::Pound
                        && (self.look_ahead(1, |t| t == &token::OpenDelim(Delimiter::Bracket))))
                {
                    // This is likely another field, TokenKind::Pound is used for `#[..]`
                    // attribute for next field. Emit the diagnostic and continue parsing.
                    err.span_suggestion(
                        sp,
                        "try adding a comma",
                        ",",
                        Applicability::MachineApplicable,
                    );
                    err.emit();
                } else {
                    return Err(err);
                }
            }
        }
        Ok(a_var)
    }

    fn expect_field_ty_separator(&mut self) -> PResult<'a, ()> {
        if let Err(err) = self.expect(exp!(Colon)) {
            let sm = self.psess.source_map();
            let eq_typo = self.token == token::Eq && self.look_ahead(1, |t| t.is_path_start());
            let semi_typo = self.token == token::Semi
                && self.look_ahead(1, |t| {
                    t.is_path_start()
                    // We check that we are in a situation like `foo; bar` to avoid bad suggestions
                    // when there's no type and `;` was used instead of a comma.
                    && match (sm.lookup_line(self.token.span.hi()), sm.lookup_line(t.span.lo())) {
                        (Ok(l), Ok(r)) => l.line == r.line,
                        _ => true,
                    }
                });
            if eq_typo || semi_typo {
                self.bump();
                // Gracefully handle small typos.
                err.with_span_suggestion_short(
                    self.prev_token.span,
                    "field names and their types are separated with `:`",
                    ":",
                    Applicability::MachineApplicable,
                )
                .emit();
            } else {
                return Err(err);
            }
        }
        Ok(())
    }

    /// Parses a structure field.
    fn parse_name_and_ty(
        &mut self,
        adt_ty: &str,
        lo: Span,
        vis: Visibility,
        safety: Safety,
        attrs: AttrVec,
    ) -> PResult<'a, FieldDef> {
        let name = self.parse_field_ident(adt_ty, lo)?;
        if self.token == token::Bang {
            if let Err(mut err) = self.unexpected() {
                // Encounter the macro invocation
                err.subdiagnostic(MacroExpandsToAdtField { adt_ty });
                return Err(err);
            }
        }
        self.expect_field_ty_separator()?;
        let ty = self.parse_ty()?;
        let default = if self.token == token::Eq {
            self.bump();
            let const_expr = self.parse_expr_anon_const()?;
            let sp = ty.span.shrink_to_hi().to(const_expr.value.span);
            self.psess.gated_spans.gate(sym::default_field_values, sp);
            Some(const_expr)
        } else {
            None
        };
        Ok(FieldDef {
            span: lo.to(self.prev_token.span),
            ident: Some(name),
            vis,
            safety,
            id: DUMMY_NODE_ID,
            ty,
            default,
            attrs,
            is_placeholder: false,
        })
    }

    /// Parses a field identifier. Specialized version of `parse_ident_common`
    /// for better diagnostics and suggestions.
    fn parse_field_ident(&mut self, adt_ty: &str, lo: Span) -> PResult<'a, Ident> {
        let (ident, is_raw) = self.ident_or_err(true)?;
        if matches!(is_raw, IdentIsRaw::No) && ident.is_reserved() {
            let snapshot = self.create_snapshot_for_diagnostic();
            let err = if self.check_fn_front_matter(false, Case::Sensitive) {
                let inherited_vis =
                    Visibility { span: DUMMY_SP, kind: VisibilityKind::Inherited, tokens: None };
                // We use `parse_fn` to get a span for the function
                let fn_parse_mode = FnParseMode { req_name: |_| true, req_body: true };
                match self.parse_fn(
                    &mut AttrVec::new(),
                    fn_parse_mode,
                    lo,
                    &inherited_vis,
                    Case::Insensitive,
                ) {
                    Ok(_) => {
                        self.dcx().struct_span_err(
                            lo.to(self.prev_token.span),
                            format!("functions are not allowed in {adt_ty} definitions"),
                        )
                        .with_help(
                            "unlike in C++, Java, and C#, functions are declared in `impl` blocks",
                        )
                        .with_help("see https://doc.rust-lang.org/book/ch05-03-method-syntax.html for more information")
                    }
                    Err(err) => {
                        err.cancel();
                        self.restore_snapshot(snapshot);
                        self.expected_ident_found_err()
                    }
                }
            } else if self.eat_keyword(exp!(Struct)) {
                match self.parse_item_struct() {
                    Ok(item) => {
                        let ItemKind::Struct(ident, ..) = item else { unreachable!() };
                        self.dcx()
                            .struct_span_err(
                                lo.with_hi(ident.span.hi()),
                                format!("structs are not allowed in {adt_ty} definitions"),
                            )
                            .with_help(
                                "consider creating a new `struct` definition instead of nesting",
                            )
                    }
                    Err(err) => {
                        err.cancel();
                        self.restore_snapshot(snapshot);
                        self.expected_ident_found_err()
                    }
                }
            } else {
                let mut err = self.expected_ident_found_err();
                if self.eat_keyword_noexpect(kw::Let)
                    && let removal_span = self.prev_token.span.until(self.token.span)
                    && let Ok(ident) = self
                        .parse_ident_common(false)
                        // Cancel this error, we don't need it.
                        .map_err(|err| err.cancel())
                    && self.token == TokenKind::Colon
                {
                    err.span_suggestion(
                        removal_span,
                        "remove this `let` keyword",
                        String::new(),
                        Applicability::MachineApplicable,
                    );
                    err.note("the `let` keyword is not allowed in `struct` fields");
                    err.note("see <https://doc.rust-lang.org/book/ch05-01-defining-structs.html> for more information");
                    err.emit();
                    return Ok(ident);
                } else {
                    self.restore_snapshot(snapshot);
                }
                err
            };
            return Err(err);
        }
        self.bump();
        Ok(ident)
    }

    /// Parses a declarative macro 2.0 definition.
    /// The `macro` keyword has already been parsed.
    /// ```ebnf
    /// MacBody = "{" TOKEN_STREAM "}" ;
    /// MacParams = "(" TOKEN_STREAM ")" ;
    /// DeclMac = "macro" Ident MacParams? MacBody ;
    /// ```
    fn parse_item_decl_macro(&mut self, lo: Span) -> PResult<'a, ItemKind> {
        let ident = self.parse_ident()?;
        let body = if self.check(exp!(OpenBrace)) {
            self.parse_delim_args()? // `MacBody`
        } else if self.check(exp!(OpenParen)) {
            let params = self.parse_token_tree(); // `MacParams`
            let pspan = params.span();
            if !self.check(exp!(OpenBrace)) {
                self.unexpected()?;
            }
            let body = self.parse_token_tree(); // `MacBody`
            // Convert `MacParams MacBody` into `{ MacParams => MacBody }`.
            let bspan = body.span();
            let arrow = TokenTree::token_alone(token::FatArrow, pspan.between(bspan)); // `=>`
            let tokens = TokenStream::new(vec![params, arrow, body]);
            let dspan = DelimSpan::from_pair(pspan.shrink_to_lo(), bspan.shrink_to_hi());
            P(DelimArgs { dspan, delim: Delimiter::Brace, tokens })
        } else {
            self.unexpected_any()?
        };

        self.psess.gated_spans.gate(sym::decl_macro, lo.to(self.prev_token.span));
        Ok(ItemKind::MacroDef(ident, ast::MacroDef { body, macro_rules: false }))
    }

    /// Is this a possibly malformed start of a `macro_rules! foo` item definition?
    fn is_macro_rules_item(&mut self) -> IsMacroRulesItem {
        if self.check_keyword(exp!(MacroRules)) {
            let macro_rules_span = self.token.span;

            if self.look_ahead(1, |t| *t == token::Bang) && self.look_ahead(2, |t| t.is_ident()) {
                return IsMacroRulesItem::Yes { has_bang: true };
            } else if self.look_ahead(1, |t| (t.is_ident())) {
                // macro_rules foo
                self.dcx().emit_err(errors::MacroRulesMissingBang {
                    span: macro_rules_span,
                    hi: macro_rules_span.shrink_to_hi(),
                });

                return IsMacroRulesItem::Yes { has_bang: false };
            }
        }

        IsMacroRulesItem::No
    }

    /// Parses a `macro_rules! foo { ... }` declarative macro.
    fn parse_item_macro_rules(
        &mut self,
        vis: &Visibility,
        has_bang: bool,
    ) -> PResult<'a, ItemKind> {
        self.expect_keyword(exp!(MacroRules))?; // `macro_rules`

        if has_bang {
            self.expect(exp!(Bang))?; // `!`
        }
        let ident = self.parse_ident()?;

        if self.eat(exp!(Bang)) {
            // Handle macro_rules! foo!
            let span = self.prev_token.span;
            self.dcx().emit_err(errors::MacroNameRemoveBang { span });
        }

        let body = self.parse_delim_args()?;
        self.eat_semi_for_macro_if_needed(&body);
        self.complain_if_pub_macro(vis, true);

        Ok(ItemKind::MacroDef(ident, ast::MacroDef { body, macro_rules: true }))
    }

    /// Item macro invocations or `macro_rules!` definitions need inherited visibility.
    /// If that's not the case, emit an error.
    fn complain_if_pub_macro(&self, vis: &Visibility, macro_rules: bool) {
        if let VisibilityKind::Inherited = vis.kind {
            return;
        }

        let vstr = pprust::vis_to_string(vis);
        let vstr = vstr.trim_end();
        if macro_rules {
            self.dcx().emit_err(errors::MacroRulesVisibility { span: vis.span, vis: vstr });
        } else {
            self.dcx().emit_err(errors::MacroInvocationVisibility { span: vis.span, vis: vstr });
        }
    }

    fn eat_semi_for_macro_if_needed(&mut self, args: &DelimArgs) {
        if args.need_semicolon() && !self.eat(exp!(Semi)) {
            self.report_invalid_macro_expansion_item(args);
        }
    }

    fn report_invalid_macro_expansion_item(&self, args: &DelimArgs) {
        let span = args.dspan.entire();
        let mut err = self.dcx().struct_span_err(
            span,
            "macros that expand to items must be delimited with braces or followed by a semicolon",
        );
        // FIXME: This will make us not emit the help even for declarative
        // macros within the same crate (that we can fix), which is sad.
        if !span.from_expansion() {
            let DelimSpan { open, close } = args.dspan;
            err.multipart_suggestion(
                "change the delimiters to curly braces",
                vec![(open, "{".to_string()), (close, '}'.to_string())],
                Applicability::MaybeIncorrect,
            );
            err.span_suggestion(
                span.with_neighbor(self.token.span).shrink_to_hi(),
                "add a semicolon",
                ';',
                Applicability::MaybeIncorrect,
            );
        }
        err.emit();
    }

    /// Checks if current token is one of tokens which cannot be nested like `kw::Enum`. In case
    /// it is, we try to parse the item and report error about nested types.
    fn recover_nested_adt_item(&mut self, keyword: Symbol) -> PResult<'a, bool> {
        if (self.token.is_keyword(kw::Enum)
            || self.token.is_keyword(kw::Struct)
            || self.token.is_keyword(kw::Union))
            && self.look_ahead(1, |t| t.is_ident())
        {
            let kw_token = self.token;
            let kw_str = pprust::token_to_string(&kw_token);
            let item = self.parse_item(ForceCollect::No)?;
            let mut item = item.unwrap().span;
            if self.token == token::Comma {
                item = item.to(self.token.span);
            }
            self.dcx().emit_err(errors::NestedAdt {
                span: kw_token.span,
                item,
                kw_str,
                keyword: keyword.as_str(),
            });
            // We successfully parsed the item but we must inform the caller about nested problem.
            return Ok(false);
        }
        Ok(true)
    }
}

/// The parsing configuration used to parse a parameter list (see `parse_fn_params`).
///
/// The function decides if, per-parameter `p`, `p` must have a pattern or just a type.
///
/// This function pointer accepts an edition, because in edition 2015, trait declarations
/// were allowed to omit parameter names. In 2018, they became required.
type ReqName = fn(Edition) -> bool;

/// Parsing configuration for functions.
///
/// The syntax of function items is slightly different within trait definitions,
/// impl blocks, and modules. It is still parsed using the same code, just with
/// different flags set, so that even when the input is wrong and produces a parse
/// error, it still gets into the AST and the rest of the parser and
/// type checker can run.
#[derive(Clone, Copy)]
pub(crate) struct FnParseMode {
    /// A function pointer that decides if, per-parameter `p`, `p` must have a
    /// pattern or just a type. This field affects parsing of the parameters list.
    ///
    /// ```text
    /// fn foo(alef: A) -> X { X::new() }
    ///        -----^^ affects parsing this part of the function signature
    ///        |
    ///        if req_name returns false, then this name is optional
    ///
    /// fn bar(A) -> X;
    ///        ^
    ///        |
    ///        if req_name returns true, this is an error
    /// ```
    ///
    /// Calling this function pointer should only return false if:
    ///
    ///   * The item is being parsed inside of a trait definition.
    ///     Within an impl block or a module, it should always evaluate
    ///     to true.
    ///   * The span is from Edition 2015. In particular, you can get a
    ///     2015 span inside a 2021 crate using macros.
    pub(super) req_name: ReqName,
    /// If this flag is set to `true`, then plain, semicolon-terminated function
    /// prototypes are not allowed here.
    ///
    /// ```text
    /// fn foo(alef: A) -> X { X::new() }
    ///                      ^^^^^^^^^^^^
    ///                      |
    ///                      this is always allowed
    ///
    /// fn bar(alef: A, bet: B) -> X;
    ///                             ^
    ///                             |
    ///                             if req_body is set to true, this is an error
    /// ```
    ///
    /// This field should only be set to false if the item is inside of a trait
    /// definition or extern block. Within an impl block or a module, it should
    /// always be set to true.
    pub(super) req_body: bool,
}

/// Parsing of functions and methods.
impl<'a> Parser<'a> {
    /// Parse a function starting from the front matter (`const ...`) to the body `{ ... }` or `;`.
    fn parse_fn(
        &mut self,
        attrs: &mut AttrVec,
        fn_parse_mode: FnParseMode,
        sig_lo: Span,
        vis: &Visibility,
        case: Case,
    ) -> PResult<'a, (Ident, FnSig, Generics, Option<P<FnContract>>, Option<P<Block>>)> {
        let fn_span = self.token.span;
        let header = self.parse_fn_front_matter(vis, case)?; // `const ... fn`
        let ident = self.parse_ident()?; // `foo`
        let mut generics = self.parse_generics()?; // `<'a, T, ...>`
        let decl = match self.parse_fn_decl(
            fn_parse_mode.req_name,
            AllowPlus::Yes,
            RecoverReturnSign::Yes,
        ) {
            Ok(decl) => decl,
            Err(old_err) => {
                // If we see `for Ty ...` then user probably meant `impl` item.
                if self.token.is_keyword(kw::For) {
                    old_err.cancel();
                    return Err(self.dcx().create_err(errors::FnTypoWithImpl { fn_span }));
                } else {
                    return Err(old_err);
                }
            }
        };

        // Store the end of function parameters to give better diagnostics
        // inside `parse_fn_body()`.
        let fn_params_end = self.prev_token.span.shrink_to_hi();

        let contract = self.parse_contract()?;

        generics.where_clause = self.parse_where_clause()?; // `where T: Ord`

        // `fn_params_end` is needed only when it's followed by a where clause.
        let fn_params_end =
            if generics.where_clause.has_where_token { Some(fn_params_end) } else { None };

        let mut sig_hi = self.prev_token.span;
        // Either `;` or `{ ... }`.
        let body =
            self.parse_fn_body(attrs, &ident, &mut sig_hi, fn_parse_mode.req_body, fn_params_end)?;
        let fn_sig_span = sig_lo.to(sig_hi);
        Ok((ident, FnSig { header, decl, span: fn_sig_span }, generics, contract, body))
    }

    /// Provide diagnostics when function body is not found
    fn error_fn_body_not_found(
        &mut self,
        ident_span: Span,
        req_body: bool,
        fn_params_end: Option<Span>,
    ) -> PResult<'a, ErrorGuaranteed> {
        let expected: &[_] =
            if req_body { &[exp!(OpenBrace)] } else { &[exp!(Semi), exp!(OpenBrace)] };
        match self.expected_one_of_not_found(&[], expected) {
            Ok(error_guaranteed) => Ok(error_guaranteed),
            Err(mut err) => {
                if self.token == token::CloseDelim(Delimiter::Brace) {
                    // The enclosing `mod`, `trait` or `impl` is being closed, so keep the `fn` in
                    // the AST for typechecking.
                    err.span_label(ident_span, "while parsing this `fn`");
                    Ok(err.emit())
                } else if self.token == token::RArrow
                    && let Some(fn_params_end) = fn_params_end
                {
                    // Instead of a function body, the parser has encountered a right arrow
                    // preceded by a where clause.

                    // Find whether token behind the right arrow is a function trait and
                    // store its span.
                    let fn_trait_span =
                        [sym::FnOnce, sym::FnMut, sym::Fn].into_iter().find_map(|symbol| {
                            if self.prev_token.is_ident_named(symbol) {
                                Some(self.prev_token.span)
                            } else {
                                None
                            }
                        });

                    // Parse the return type (along with the right arrow) and store its span.
                    // If there's a parse error, cancel it and return the existing error
                    // as we are primarily concerned with the
                    // expected-function-body-but-found-something-else error here.
                    let arrow_span = self.token.span;
                    let ty_span = match self.parse_ret_ty(
                        AllowPlus::Yes,
                        RecoverQPath::Yes,
                        RecoverReturnSign::Yes,
                    ) {
                        Ok(ty_span) => ty_span.span().shrink_to_hi(),
                        Err(parse_error) => {
                            parse_error.cancel();
                            return Err(err);
                        }
                    };
                    let ret_ty_span = arrow_span.to(ty_span);

                    if let Some(fn_trait_span) = fn_trait_span {
                        // Typo'd Fn* trait bounds such as
                        // fn foo<F>() where F: FnOnce -> () {}
                        err.subdiagnostic(errors::FnTraitMissingParen { span: fn_trait_span });
                    } else if let Ok(snippet) = self.psess.source_map().span_to_snippet(ret_ty_span)
                    {
                        // If token behind right arrow is not a Fn* trait, the programmer
                        // probably misplaced the return type after the where clause like
                        // `fn foo<T>() where T: Default -> u8 {}`
                        err.primary_message(
                            "return type should be specified after the function parameters",
                        );
                        err.subdiagnostic(errors::MisplacedReturnType {
                            fn_params_end,
                            snippet,
                            ret_ty_span,
                        });
                    }
                    Err(err)
                } else {
                    Err(err)
                }
            }
        }
    }

    /// Parse the "body" of a function.
    /// This can either be `;` when there's no body,
    /// or e.g. a block when the function is a provided one.
    fn parse_fn_body(
        &mut self,
        attrs: &mut AttrVec,
        ident: &Ident,
        sig_hi: &mut Span,
        req_body: bool,
        fn_params_end: Option<Span>,
    ) -> PResult<'a, Option<P<Block>>> {
        let has_semi = if req_body {
            self.token == TokenKind::Semi
        } else {
            // Only include `;` in list of expected tokens if body is not required
            self.check(exp!(Semi))
        };
        let (inner_attrs, body) = if has_semi {
            // Include the trailing semicolon in the span of the signature
            self.expect_semi()?;
            *sig_hi = self.prev_token.span;
            (AttrVec::new(), None)
        } else if self.check(exp!(OpenBrace)) || self.token.is_metavar_block() {
            self.parse_block_common(self.token.span, BlockCheckMode::Default, None)
                .map(|(attrs, body)| (attrs, Some(body)))?
        } else if self.token == token::Eq {
            // Recover `fn foo() = $expr;`.
            self.bump(); // `=`
            let eq_sp = self.prev_token.span;
            let _ = self.parse_expr()?;
            self.expect_semi()?; // `;`
            let span = eq_sp.to(self.prev_token.span);
            let guar = self.dcx().emit_err(errors::FunctionBodyEqualsExpr {
                span,
                sugg: errors::FunctionBodyEqualsExprSugg { eq: eq_sp, semi: self.prev_token.span },
            });
            (AttrVec::new(), Some(self.mk_block_err(span, guar)))
        } else {
            self.error_fn_body_not_found(ident.span, req_body, fn_params_end)?;
            (AttrVec::new(), None)
        };
        attrs.extend(inner_attrs);
        Ok(body)
    }

    /// Is the current token the start of an `FnHeader` / not a valid parse?
    ///
    /// `check_pub` adds additional `pub` to the checks in case users place it
    /// wrongly, can be used to ensure `pub` never comes after `default`.
    pub(super) fn check_fn_front_matter(&mut self, check_pub: bool, case: Case) -> bool {
        const ALL_QUALS: &[ExpKeywordPair] = &[
            exp!(Pub),
            exp!(Gen),
            exp!(Const),
            exp!(Async),
            exp!(Unsafe),
            exp!(Safe),
            exp!(Extern),
        ];

        // We use an over-approximation here.
        // `const const`, `fn const` won't parse, but we're not stepping over other syntax either.
        // `pub` is added in case users got confused with the ordering like `async pub fn`,
        // only if it wasn't preceded by `default` as `default pub` is invalid.
        let quals: &[_] = if check_pub {
            ALL_QUALS
        } else {
            &[exp!(Gen), exp!(Const), exp!(Async), exp!(Unsafe), exp!(Safe), exp!(Extern)]
        };
        self.check_keyword_case(exp!(Fn), case) // Definitely an `fn`.
            // `$qual fn` or `$qual $qual`:
            || quals.iter().any(|&exp| self.check_keyword_case(exp, case))
                && self.look_ahead(1, |t| {
                    // `$qual fn`, e.g. `const fn` or `async fn`.
                    t.is_keyword_case(kw::Fn, case)
                    // Two qualifiers `$qual $qual` is enough, e.g. `async unsafe`.
                    || (
                        (
                            t.is_non_raw_ident_where(|i|
                                quals.iter().any(|exp| exp.kw == i.name)
                                    // Rule out 2015 `const async: T = val`.
                                    && i.is_reserved()
                            )
                            || case == Case::Insensitive
                                && t.is_non_raw_ident_where(|i| quals.iter().any(|exp| {
                                    exp.kw.as_str() == i.name.as_str().to_lowercase()
                                }))
                        )
                        // Rule out `unsafe extern {`.
                        && !self.is_unsafe_foreign_mod()
                        // Rule out `async gen {` and `async gen move {`
                        && !self.is_async_gen_block())
                })
            // `extern ABI fn`
            || self.check_keyword_case(exp!(Extern), case)
                // Use `tree_look_ahead` because `ABI` might be a metavariable,
                // i.e. an invisible-delimited sequence, and `tree_look_ahead`
                // will consider that a single element when looking ahead.
                && self.look_ahead(1, |t| t.can_begin_string_literal())
                && (self.tree_look_ahead(2, |tt| {
                    match tt {
                        TokenTree::Token(t, _) => t.is_keyword_case(kw::Fn, case),
                        TokenTree::Delimited(..) => false,
                    }
                }) == Some(true) ||
                    // This branch is only for better diagnostics; `pub`, `unsafe`, etc. are not
                    // allowed here.
                    (self.may_recover()
                        && self.tree_look_ahead(2, |tt| {
                            match tt {
                                TokenTree::Token(t, _) =>
                                    ALL_QUALS.iter().any(|exp| {
                                        t.is_keyword(exp.kw)
                                    }),
                                TokenTree::Delimited(..) => false,
                            }
                        }) == Some(true)
                        && self.tree_look_ahead(3, |tt| {
                            match tt {
                                TokenTree::Token(t, _) => t.is_keyword_case(kw::Fn, case),
                                TokenTree::Delimited(..) => false,
                            }
                        }) == Some(true)
                    )
                )
    }

    /// Parses all the "front matter" (or "qualifiers") for a `fn` declaration,
    /// up to and including the `fn` keyword. The formal grammar is:
    ///
    /// ```text
    /// Extern = "extern" StringLit? ;
    /// FnQual = "const"? "async"? "unsafe"? Extern? ;
    /// FnFrontMatter = FnQual "fn" ;
    /// ```
    ///
    /// `vis` represents the visibility that was already parsed, if any. Use
    /// `Visibility::Inherited` when no visibility is known.
    pub(super) fn parse_fn_front_matter(
        &mut self,
        orig_vis: &Visibility,
        case: Case,
    ) -> PResult<'a, FnHeader> {
        let sp_start = self.token.span;
        let constness = self.parse_constness(case);

        let async_start_sp = self.token.span;
        let coroutine_kind = self.parse_coroutine_kind(case);

        let unsafe_start_sp = self.token.span;
        let safety = self.parse_safety(case);

        let ext_start_sp = self.token.span;
        let ext = self.parse_extern(case);

        if let Some(CoroutineKind::Async { span, .. }) = coroutine_kind {
            if span.is_rust_2015() {
                self.dcx().emit_err(errors::AsyncFnIn2015 {
                    span,
                    help: errors::HelpUseLatestEdition::new(),
                });
            }
        }

        match coroutine_kind {
            Some(CoroutineKind::Gen { span, .. }) | Some(CoroutineKind::AsyncGen { span, .. }) => {
                self.psess.gated_spans.gate(sym::gen_blocks, span);
            }
            Some(CoroutineKind::Async { .. }) | None => {}
        }

        if !self.eat_keyword_case(exp!(Fn), case) {
            // It is possible for `expect_one_of` to recover given the contents of
            // `self.expected_token_types`, therefore, do not use `self.unexpected()` which doesn't
            // account for this.
            match self.expect_one_of(&[], &[]) {
                Ok(Recovered::Yes(_)) => {}
                Ok(Recovered::No) => unreachable!(),
                Err(mut err) => {
                    // Qualifier keywords ordering check
                    enum WrongKw {
                        Duplicated(Span),
                        Misplaced(Span),
                    }

                    // We may be able to recover
                    let mut recover_constness = constness;
                    let mut recover_coroutine_kind = coroutine_kind;
                    let mut recover_safety = safety;
                    // This will allow the machine fix to directly place the keyword in the correct place or to indicate
                    // that the keyword is already present and the second instance should be removed.
                    let wrong_kw = if self.check_keyword(exp!(Const)) {
                        match constness {
                            Const::Yes(sp) => Some(WrongKw::Duplicated(sp)),
                            Const::No => {
                                recover_constness = Const::Yes(self.token.span);
                                Some(WrongKw::Misplaced(async_start_sp))
                            }
                        }
                    } else if self.check_keyword(exp!(Async)) {
                        match coroutine_kind {
                            Some(CoroutineKind::Async { span, .. }) => {
                                Some(WrongKw::Duplicated(span))
                            }
                            Some(CoroutineKind::AsyncGen { span, .. }) => {
                                Some(WrongKw::Duplicated(span))
                            }
                            Some(CoroutineKind::Gen { .. }) => {
                                recover_coroutine_kind = Some(CoroutineKind::AsyncGen {
                                    span: self.token.span,
                                    closure_id: DUMMY_NODE_ID,
                                    return_impl_trait_id: DUMMY_NODE_ID,
                                });
                                // FIXME(gen_blocks): This span is wrong, didn't want to think about it.
                                Some(WrongKw::Misplaced(unsafe_start_sp))
                            }
                            None => {
                                recover_coroutine_kind = Some(CoroutineKind::Async {
                                    span: self.token.span,
                                    closure_id: DUMMY_NODE_ID,
                                    return_impl_trait_id: DUMMY_NODE_ID,
                                });
                                Some(WrongKw::Misplaced(unsafe_start_sp))
                            }
                        }
                    } else if self.check_keyword(exp!(Unsafe)) {
                        match safety {
                            Safety::Unsafe(sp) => Some(WrongKw::Duplicated(sp)),
                            Safety::Safe(sp) => {
                                recover_safety = Safety::Unsafe(self.token.span);
                                Some(WrongKw::Misplaced(sp))
                            }
                            Safety::Default => {
                                recover_safety = Safety::Unsafe(self.token.span);
                                Some(WrongKw::Misplaced(ext_start_sp))
                            }
                        }
                    } else if self.check_keyword(exp!(Safe)) {
                        match safety {
                            Safety::Safe(sp) => Some(WrongKw::Duplicated(sp)),
                            Safety::Unsafe(sp) => {
                                recover_safety = Safety::Safe(self.token.span);
                                Some(WrongKw::Misplaced(sp))
                            }
                            Safety::Default => {
                                recover_safety = Safety::Safe(self.token.span);
                                Some(WrongKw::Misplaced(ext_start_sp))
                            }
                        }
                    } else {
                        None
                    };

                    // The keyword is already present, suggest removal of the second instance
                    if let Some(WrongKw::Duplicated(original_sp)) = wrong_kw {
                        let original_kw = self
                            .span_to_snippet(original_sp)
                            .expect("Span extracted directly from keyword should always work");

                        err.span_suggestion(
                            self.token_uninterpolated_span(),
                            format!("`{original_kw}` already used earlier, remove this one"),
                            "",
                            Applicability::MachineApplicable,
                        )
                        .span_note(original_sp, format!("`{original_kw}` first seen here"));
                    }
                    // The keyword has not been seen yet, suggest correct placement in the function front matter
                    else if let Some(WrongKw::Misplaced(correct_pos_sp)) = wrong_kw {
                        let correct_pos_sp = correct_pos_sp.to(self.prev_token.span);
                        if let Ok(current_qual) = self.span_to_snippet(correct_pos_sp) {
                            let misplaced_qual_sp = self.token_uninterpolated_span();
                            let misplaced_qual = self.span_to_snippet(misplaced_qual_sp).unwrap();

                            err.span_suggestion(
                                    correct_pos_sp.to(misplaced_qual_sp),
                                    format!("`{misplaced_qual}` must come before `{current_qual}`"),
                                    format!("{misplaced_qual} {current_qual}"),
                                    Applicability::MachineApplicable,
                                ).note("keyword order for functions declaration is `pub`, `default`, `const`, `async`, `unsafe`, `extern`");
                        }
                    }
                    // Recover incorrect visibility order such as `async pub`
                    else if self.check_keyword(exp!(Pub)) {
                        let sp = sp_start.to(self.prev_token.span);
                        if let Ok(snippet) = self.span_to_snippet(sp) {
                            let current_vis = match self.parse_visibility(FollowedByType::No) {
                                Ok(v) => v,
                                Err(d) => {
                                    d.cancel();
                                    return Err(err);
                                }
                            };
                            let vs = pprust::vis_to_string(&current_vis);
                            let vs = vs.trim_end();

                            // There was no explicit visibility
                            if matches!(orig_vis.kind, VisibilityKind::Inherited) {
                                err.span_suggestion(
                                    sp_start.to(self.prev_token.span),
                                    format!("visibility `{vs}` must come before `{snippet}`"),
                                    format!("{vs} {snippet}"),
                                    Applicability::MachineApplicable,
                                );
                            }
                            // There was an explicit visibility
                            else {
                                err.span_suggestion(
                                    current_vis.span,
                                    "there is already a visibility modifier, remove one",
                                    "",
                                    Applicability::MachineApplicable,
                                )
                                .span_note(orig_vis.span, "explicit visibility first seen here");
                            }
                        }
                    }

                    // FIXME(gen_blocks): add keyword recovery logic for genness

                    if wrong_kw.is_some()
                        && self.may_recover()
                        && self.look_ahead(1, |tok| tok.is_keyword_case(kw::Fn, case))
                    {
                        // Advance past the misplaced keyword and `fn`
                        self.bump();
                        self.bump();
                        err.emit();
                        return Ok(FnHeader {
                            constness: recover_constness,
                            safety: recover_safety,
                            coroutine_kind: recover_coroutine_kind,
                            ext,
                        });
                    }

                    return Err(err);
                }
            }
        }

        Ok(FnHeader { constness, safety, coroutine_kind, ext })
    }

    /// Parses the parameter list and result type of a function declaration.
    pub(super) fn parse_fn_decl(
        &mut self,
        req_name: ReqName,
        ret_allow_plus: AllowPlus,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, P<FnDecl>> {
        Ok(P(FnDecl {
            inputs: self.parse_fn_params(req_name)?,
            output: self.parse_ret_ty(ret_allow_plus, RecoverQPath::Yes, recover_return_sign)?,
        }))
    }

    /// Parses the parameter list of a function, including the `(` and `)` delimiters.
    pub(super) fn parse_fn_params(&mut self, req_name: ReqName) -> PResult<'a, ThinVec<Param>> {
        let mut first_param = true;
        // Parse the arguments, starting out with `self` being allowed...
        if self.token != TokenKind::OpenDelim(Delimiter::Parenthesis)
        // might be typo'd trait impl, handled elsewhere
        && !self.token.is_keyword(kw::For)
        {
            // recover from missing argument list, e.g. `fn main -> () {}`
            self.dcx()
                .emit_err(errors::MissingFnParams { span: self.prev_token.span.shrink_to_hi() });
            return Ok(ThinVec::new());
        }

        let (mut params, _) = self.parse_paren_comma_seq(|p| {
            p.recover_vcs_conflict_marker();
            let snapshot = p.create_snapshot_for_diagnostic();
            let param = p.parse_param_general(req_name, first_param).or_else(|e| {
                let guar = e.emit();
                // When parsing a param failed, we should check to make the span of the param
                // not contain '(' before it.
                // For example when parsing `*mut Self` in function `fn oof(*mut Self)`.
                let lo = if let TokenKind::OpenDelim(Delimiter::Parenthesis) = p.prev_token.kind {
                    p.prev_token.span.shrink_to_hi()
                } else {
                    p.prev_token.span
                };
                p.restore_snapshot(snapshot);
                // Skip every token until next possible arg or end.
                p.eat_to_tokens(&[exp!(Comma), exp!(CloseParen)]);
                // Create a placeholder argument for proper arg count (issue #34264).
                Ok(dummy_arg(Ident::new(sym::dummy, lo.to(p.prev_token.span)), guar))
            });
            // ...now that we've parsed the first argument, `self` is no longer allowed.
            first_param = false;
            param
        })?;
        // Replace duplicated recovered params with `_` pattern to avoid unnecessary errors.
        self.deduplicate_recovered_params_names(&mut params);
        Ok(params)
    }

    /// Parses a single function parameter.
    ///
    /// - `self` is syntactically allowed when `first_param` holds.
    fn parse_param_general(&mut self, req_name: ReqName, first_param: bool) -> PResult<'a, Param> {
        let lo = self.token.span;
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            // Possibly parse `self`. Recover if we parsed it and it wasn't allowed here.
            if let Some(mut param) = this.parse_self_param()? {
                param.attrs = attrs;
                let res = if first_param { Ok(param) } else { this.recover_bad_self_param(param) };
                return Ok((res?, Trailing::No, UsePreAttrPos::No));
            }

            let is_name_required = match this.token.kind {
                token::DotDotDot => false,
                _ => req_name(this.token.span.with_neighbor(this.prev_token.span).edition()),
            };
            let (pat, ty) = if is_name_required || this.is_named_param() {
                debug!("parse_param_general parse_pat (is_name_required:{})", is_name_required);
                let (pat, colon) = this.parse_fn_param_pat_colon()?;
                if !colon {
                    let mut err = this.unexpected().unwrap_err();
                    return if let Some(ident) =
                        this.parameter_without_type(&mut err, pat, is_name_required, first_param)
                    {
                        let guar = err.emit();
                        Ok((dummy_arg(ident, guar), Trailing::No, UsePreAttrPos::No))
                    } else {
                        Err(err)
                    };
                }

                this.eat_incorrect_doc_comment_for_param_type();
                (pat, this.parse_ty_for_param()?)
            } else {
                debug!("parse_param_general ident_to_pat");
                let parser_snapshot_before_ty = this.create_snapshot_for_diagnostic();
                this.eat_incorrect_doc_comment_for_param_type();
                let mut ty = this.parse_ty_for_param();

                if let Ok(t) = &ty {
                    // Check for trailing angle brackets
                    if let TyKind::Path(_, Path { segments, .. }) = &t.kind {
                        if let Some(segment) = segments.last() {
                            if let Some(guar) =
                                this.check_trailing_angle_brackets(segment, &[exp!(CloseParen)])
                            {
                                return Ok((
                                    dummy_arg(segment.ident, guar),
                                    Trailing::No,
                                    UsePreAttrPos::No,
                                ));
                            }
                        }
                    }

                    if this.token != token::Comma
                        && this.token != token::CloseDelim(Delimiter::Parenthesis)
                    {
                        // This wasn't actually a type, but a pattern looking like a type,
                        // so we are going to rollback and re-parse for recovery.
                        ty = this.unexpected_any();
                    }
                }
                match ty {
                    Ok(ty) => {
                        let pat = this.mk_pat(ty.span, PatKind::Missing);
                        (pat, ty)
                    }
                    // If this is a C-variadic argument and we hit an error, return the error.
                    Err(err) if this.token == token::DotDotDot => return Err(err),
                    Err(err) if this.unmatched_angle_bracket_count > 0 => return Err(err),
                    // Recover from attempting to parse the argument as a type without pattern.
                    Err(err) => {
                        err.cancel();
                        this.restore_snapshot(parser_snapshot_before_ty);
                        this.recover_arg_parse()?
                    }
                }
            };

            let span = lo.to(this.prev_token.span);

            Ok((
                Param { attrs, id: ast::DUMMY_NODE_ID, is_placeholder: false, pat, span, ty },
                Trailing::No,
                UsePreAttrPos::No,
            ))
        })
    }

    /// Returns the parsed optional self parameter and whether a self shortcut was used.
    fn parse_self_param(&mut self) -> PResult<'a, Option<Param>> {
        // Extract an identifier *after* having confirmed that the token is one.
        let expect_self_ident = |this: &mut Self| match this.token.ident() {
            Some((ident, IdentIsRaw::No)) => {
                this.bump();
                ident
            }
            _ => unreachable!(),
        };
        // is lifetime `n` tokens ahead?
        let is_lifetime = |this: &Self, n| this.look_ahead(n, |t| t.is_lifetime());
        // Is `self` `n` tokens ahead?
        let is_isolated_self = |this: &Self, n| {
            this.is_keyword_ahead(n, &[kw::SelfLower])
                && this.look_ahead(n + 1, |t| t != &token::PathSep)
        };
        // Is `pin const self` `n` tokens ahead?
        let is_isolated_pin_const_self = |this: &Self, n| {
            this.look_ahead(n, |token| token.is_ident_named(sym::pin))
                && this.is_keyword_ahead(n + 1, &[kw::Const])
                && is_isolated_self(this, n + 2)
        };
        // Is `mut self` `n` tokens ahead?
        let is_isolated_mut_self =
            |this: &Self, n| this.is_keyword_ahead(n, &[kw::Mut]) && is_isolated_self(this, n + 1);
        // Is `pin mut self` `n` tokens ahead?
        let is_isolated_pin_mut_self = |this: &Self, n| {
            this.look_ahead(n, |token| token.is_ident_named(sym::pin))
                && is_isolated_mut_self(this, n + 1)
        };
        // Parse `self` or `self: TYPE`. We already know the current token is `self`.
        let parse_self_possibly_typed = |this: &mut Self, m| {
            let eself_ident = expect_self_ident(this);
            let eself_hi = this.prev_token.span;
            let eself = if this.eat(exp!(Colon)) {
                SelfKind::Explicit(this.parse_ty()?, m)
            } else {
                SelfKind::Value(m)
            };
            Ok((eself, eself_ident, eself_hi))
        };
        let expect_self_ident_not_typed =
            |this: &mut Self, modifier: &SelfKind, modifier_span: Span| {
                let eself_ident = expect_self_ident(this);

                // Recover `: Type` after a qualified self
                if this.may_recover() && this.eat_noexpect(&token::Colon) {
                    let snap = this.create_snapshot_for_diagnostic();
                    match this.parse_ty() {
                        Ok(ty) => {
                            this.dcx().emit_err(errors::IncorrectTypeOnSelf {
                                span: ty.span,
                                move_self_modifier: errors::MoveSelfModifier {
                                    removal_span: modifier_span,
                                    insertion_span: ty.span.shrink_to_lo(),
                                    modifier: modifier.to_ref_suggestion(),
                                },
                            });
                        }
                        Err(diag) => {
                            diag.cancel();
                            this.restore_snapshot(snap);
                        }
                    }
                }
                eself_ident
            };
        // Recover for the grammar `*self`, `*const self`, and `*mut self`.
        let recover_self_ptr = |this: &mut Self| {
            this.dcx().emit_err(errors::SelfArgumentPointer { span: this.token.span });

            Ok((SelfKind::Value(Mutability::Not), expect_self_ident(this), this.prev_token.span))
        };

        // Parse optional `self` parameter of a method.
        // Only a limited set of initial token sequences is considered `self` parameters; anything
        // else is parsed as a normal function parameter list, so some lookahead is required.
        let eself_lo = self.token.span;
        let (eself, eself_ident, eself_hi) = match self.token.uninterpolate().kind {
            token::And => {
                let has_lifetime = is_lifetime(self, 1);
                let skip_lifetime_count = has_lifetime as usize;
                let eself = if is_isolated_self(self, skip_lifetime_count + 1) {
                    // `&{'lt} self`
                    self.bump(); // &
                    let lifetime = has_lifetime.then(|| self.expect_lifetime());
                    SelfKind::Region(lifetime, Mutability::Not)
                } else if is_isolated_mut_self(self, skip_lifetime_count + 1) {
                    // `&{'lt} mut self`
                    self.bump(); // &
                    let lifetime = has_lifetime.then(|| self.expect_lifetime());
                    self.bump(); // mut
                    SelfKind::Region(lifetime, Mutability::Mut)
                } else if is_isolated_pin_const_self(self, skip_lifetime_count + 1) {
                    // `&{'lt} pin const self`
                    self.bump(); // &
                    let lifetime = has_lifetime.then(|| self.expect_lifetime());
                    self.psess.gated_spans.gate(sym::pin_ergonomics, self.token.span);
                    self.bump(); // pin
                    self.bump(); // const
                    SelfKind::Pinned(lifetime, Mutability::Not)
                } else if is_isolated_pin_mut_self(self, skip_lifetime_count + 1) {
                    // `&{'lt} pin mut self`
                    self.bump(); // &
                    let lifetime = has_lifetime.then(|| self.expect_lifetime());
                    self.psess.gated_spans.gate(sym::pin_ergonomics, self.token.span);
                    self.bump(); // pin
                    self.bump(); // mut
                    SelfKind::Pinned(lifetime, Mutability::Mut)
                } else {
                    // `&not_self`
                    return Ok(None);
                };
                let hi = self.token.span;
                let self_ident = expect_self_ident_not_typed(self, &eself, eself_lo.until(hi));
                (eself, self_ident, hi)
            }
            // `*self`
            token::Star if is_isolated_self(self, 1) => {
                self.bump();
                recover_self_ptr(self)?
            }
            // `*mut self` and `*const self`
            token::Star
                if self.look_ahead(1, |t| t.is_mutability()) && is_isolated_self(self, 2) =>
            {
                self.bump();
                self.bump();
                recover_self_ptr(self)?
            }
            // `self` and `self: TYPE`
            token::Ident(..) if is_isolated_self(self, 0) => {
                parse_self_possibly_typed(self, Mutability::Not)?
            }
            // `mut self` and `mut self: TYPE`
            token::Ident(..) if is_isolated_mut_self(self, 0) => {
                self.bump();
                parse_self_possibly_typed(self, Mutability::Mut)?
            }
            _ => return Ok(None),
        };

        let eself = source_map::respan(eself_lo.to(eself_hi), eself);
        Ok(Some(Param::from_self(AttrVec::default(), eself, eself_ident)))
    }

    fn is_named_param(&self) -> bool {
        let offset = match &self.token.kind {
            token::OpenDelim(Delimiter::Invisible(origin)) => match origin {
                InvisibleOrigin::MetaVar(MetaVarKind::Pat(_)) => {
                    return self.check_noexpect_past_close_delim(&token::Colon);
                }
                _ => 0,
            },
            token::And | token::AndAnd => 1,
            _ if self.token.is_keyword(kw::Mut) => 1,
            _ => 0,
        };

        self.look_ahead(offset, |t| t.is_ident())
            && self.look_ahead(offset + 1, |t| t == &token::Colon)
    }

    fn recover_self_param(&mut self) -> bool {
        matches!(
            self.parse_outer_attributes()
                .and_then(|_| self.parse_self_param())
                .map_err(|e| e.cancel()),
            Ok(Some(_))
        )
    }
}

enum IsMacroRulesItem {
    Yes { has_bang: bool },
    No,
}
