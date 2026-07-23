use ast::token::IdentIsRaw;
use rustc_ast as ast;
use rustc_ast::ast::*;
use rustc_ast::token::{self, InvisibleOrigin, MetaVarKind, TokenKind};
use rustc_ast::tokenstream::TokenTree;
use rustc_ast::util::case::Case;
use rustc_ast_pretty::pprust;
use rustc_errors::{Applicability, PResult};
use rustc_session::lint::builtin::VARARGS_WITHOUT_PATTERN;
use rustc_span::edition::Edition;
use rustc_span::{ErrorGuaranteed, Ident, Span, kw, respan, sym};
use thin_vec::ThinVec;
use tracing::debug;

use super::diagnostics::dummy_arg;
use super::ty::{AllowPlus, RecoverQPath, RecoverReturnSign};
use super::{
    ExpKeywordPair, FollowedByType, ForceCollect, Parser, Recovered, Trailing, UsePreAttrPos,
};
use crate::diagnostics::{self, FnPointerCannotBeAsync, FnPointerCannotBeConst};
use crate::exp;

/// The parsing configuration used to parse a parameter list (see `parse_fn_params`).
///
/// The function decides if, per-parameter `p`, `p` must have a pattern or just a type.
///
/// This function pointer accepts an edition, because in edition 2015, trait declarations
/// were allowed to omit parameter names. In 2018, they became required. It also accepts an
/// `IsDotDotDot` parameter, as `extern` function declarations and function pointer types are
/// allowed to omit the name of the `...` but regular function items are not.
type ReqName = fn(Edition, IsDotDotDot) -> bool;

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum IsDotDotDot {
    Yes,
    No,
}

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
    ///
    /// Or if `IsDotDotDot::Yes`, this function will also return `false` if the item being parsed
    /// is inside an `extern` block.
    pub(super) req_name: ReqName,
    /// The context in which this function is parsed, used for diagnostics.
    /// This indicates the fn is a free function or method and so on.
    pub(super) context: FnContext,
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

/// The context in which a function is parsed.
/// FIXME(estebank, xizheyin): Use more variants.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FnContext {
    /// Free context.
    Free,
    /// A Trait context.
    Trait,
    /// An Impl block.
    Impl,
}

/// Parsing of functions and methods.
impl<'a> Parser<'a> {
    /// Parse a function starting from the front matter (`const ...`) to the body `{ ... }` or `;`.
    pub(crate) fn parse_fn(
        &mut self,
        attrs: &mut AttrVec,
        fn_parse_mode: FnParseMode,
        sig_lo: Span,
        vis: &Visibility,
        case: Case,
    ) -> PResult<'a, (Ident, FnSig, Generics, Option<Box<FnContract>>, Option<Box<Block>>)> {
        let fn_span = self.token.span;
        let header = self.parse_fn_front_matter(vis, case, FrontMatterParsingMode::Function)?; // `const ... fn`
        let ident = self.parse_ident()?; // `foo`
        let mut generics = self.parse_generics()?; // `<'a, T, ...>`
        let decl = match self.parse_fn_decl(&fn_parse_mode, AllowPlus::Yes, RecoverReturnSign::Yes)
        {
            Ok(decl) => decl,
            Err(old_err) => {
                // If we see `for Ty ...` then user probably meant `impl` item.
                if self.token.is_keyword(kw::For) {
                    old_err.cancel();
                    return Err(self.dcx().create_err(diagnostics::FnTypoWithImpl { fn_span }));
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
                if self.token == token::CloseBrace {
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
                        err.subdiagnostic(diagnostics::FnTraitMissingParen { span: fn_trait_span });
                    } else if let Ok(snippet) = self.psess.source_map().span_to_snippet(ret_ty_span)
                    {
                        // If token behind right arrow is not a Fn* trait, the programmer
                        // probably misplaced the return type after the where clause like
                        // `fn foo<T>() where T: Default -> u8 {}`
                        err.primary_message(
                            "return type should be specified after the function parameters",
                        );
                        err.subdiagnostic(diagnostics::MisplacedReturnType {
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
    ) -> PResult<'a, Option<Box<Block>>> {
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
            let prev_in_fn_body = self.in_fn_body;
            self.in_fn_body = true;
            let res = self.parse_block_common(self.token.span, BlockCheckMode::Default, None).map(
                |(attrs, mut body)| {
                    if let Some(guar) = self.fn_body_missing_semi_guar.take() {
                        body.stmts.push(self.mk_stmt(
                            body.span,
                            StmtKind::Expr(self.mk_expr(body.span, ExprKind::Err(guar))),
                        ));
                    }
                    (attrs, Some(body))
                },
            );
            self.in_fn_body = prev_in_fn_body;
            res?
        } else if self.token == token::Eq {
            // Recover `fn foo() = $expr;`.
            self.bump(); // `=`
            let eq_sp = self.prev_token.span;
            let _ = self.parse_expr()?;
            self.expect_semi()?; // `;`
            let span = eq_sp.to(self.prev_token.span);
            let guar = self.dcx().emit_err(diagnostics::FunctionBodyEqualsExpr {
                span,
                sugg: diagnostics::FunctionBodyEqualsExprSugg {
                    eq: eq_sp,
                    semi: self.prev_token.span,
                },
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
                        && !self.is_async_gen_block()
                        // Rule out `const unsafe auto` and `const unsafe trait` and `const unsafe impl`
                        && !self.is_keyword_ahead(2, &[kw::Auto, kw::Trait, kw::Impl])
                    )
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
    ///
    /// If `parsing_mode` is `FrontMatterParsingMode::FunctionPtrType`, we error on `const` and `async` qualifiers,
    /// which are not allowed in function pointer types.
    pub(super) fn parse_fn_front_matter(
        &mut self,
        orig_vis: &Visibility,
        case: Case,
        parsing_mode: FrontMatterParsingMode,
    ) -> PResult<'a, FnHeader> {
        let sp_start = self.token.span;
        let constness = self.parse_constness(case);
        if parsing_mode == FrontMatterParsingMode::FunctionPtrType
            && let Const::Yes(const_span) = constness
        {
            self.dcx().emit_err(FnPointerCannotBeConst {
                span: const_span,
                suggestion: const_span.until(self.token.span),
            });
        }

        let async_start_sp = self.token.span;
        let coroutine_kind = self.parse_coroutine_kind(case);
        if parsing_mode == FrontMatterParsingMode::FunctionPtrType
            && let Some(ast::CoroutineKind::Async { span: async_span, .. }) = coroutine_kind
        {
            self.dcx().emit_err(FnPointerCannotBeAsync {
                span: async_span,
                suggestion: async_span.until(self.token.span),
            });
        }
        // FIXME(gen_blocks): emit a similar error for `gen fn()`

        let unsafe_start_sp = self.token.span;
        let safety = self.parse_safety(case);

        let ext_start_sp = self.token.span;
        let ext = self.parse_extern(case);

        if let Some(CoroutineKind::Async { span, .. }) = coroutine_kind {
            if span.is_rust_2015() {
                self.dcx().emit_err(diagnostics::AsyncFnIn2015 {
                    span,
                    help: diagnostics::HelpUseLatestEdition::new(),
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
                        /// `MisplacedDisallowedQualifier` is only used instead of `Misplaced`,
                        /// when the misplaced keyword is disallowed by the current `FrontMatterParsingMode`.
                        /// In this case, we avoid generating the suggestion to swap around the keywords,
                        /// as we already generated a suggestion to remove the keyword earlier.
                        MisplacedDisallowedQualifier,
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
                                match parsing_mode {
                                    FrontMatterParsingMode::Function => {
                                        Some(WrongKw::Misplaced(async_start_sp))
                                    }
                                    FrontMatterParsingMode::FunctionPtrType => {
                                        self.dcx().emit_err(FnPointerCannotBeConst {
                                            span: self.token.span,
                                            suggestion: self
                                                .token
                                                .span
                                                .with_lo(self.prev_token.span.hi()),
                                        });
                                        Some(WrongKw::MisplacedDisallowedQualifier)
                                    }
                                }
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
                                match parsing_mode {
                                    FrontMatterParsingMode::Function => {
                                        Some(WrongKw::Misplaced(async_start_sp))
                                    }
                                    FrontMatterParsingMode::FunctionPtrType => {
                                        self.dcx().emit_err(FnPointerCannotBeAsync {
                                            span: self.token.span,
                                            suggestion: self
                                                .token
                                                .span
                                                .with_lo(self.prev_token.span.hi()),
                                        });
                                        Some(WrongKw::MisplacedDisallowedQualifier)
                                    }
                                }
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

                    if let Some(wrong_kw) = wrong_kw
                        && self.may_recover()
                        && self.look_ahead(1, |tok| tok.is_keyword_case(kw::Fn, case))
                    {
                        // Advance past the misplaced keyword and `fn`
                        self.bump();
                        self.bump();
                        // When we recover from a `MisplacedDisallowedQualifier`, we already emitted an error for the disallowed qualifier
                        // So we don't emit another error that the qualifier is unexpected.
                        if matches!(wrong_kw, WrongKw::MisplacedDisallowedQualifier) {
                            err.cancel();
                        } else {
                            err.emit();
                        }
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
        fn_parse_mode: &FnParseMode,
        ret_allow_plus: AllowPlus,
        recover_return_sign: RecoverReturnSign,
    ) -> PResult<'a, Box<FnDecl>> {
        Ok(Box::new(FnDecl {
            inputs: self.parse_fn_params(fn_parse_mode)?,
            output: self.parse_ret_ty(ret_allow_plus, RecoverQPath::Yes, recover_return_sign)?,
        }))
    }

    /// Parses the parameter list of a function, including the `(` and `)` delimiters.
    pub(super) fn parse_fn_params(
        &mut self,
        fn_parse_mode: &FnParseMode,
    ) -> PResult<'a, ThinVec<Param>> {
        let mut first_param = true;
        // Parse the arguments, starting out with `self` being allowed...
        if self.token != TokenKind::OpenParen
        // might be typo'd trait impl, handled elsewhere
        && !self.token.is_keyword(kw::For)
        {
            // recover from missing argument list, e.g. `fn main -> () {}`
            self.dcx().emit_err(diagnostics::MissingFnParams {
                span: self.prev_token.span.shrink_to_hi(),
            });
            return Ok(ThinVec::new());
        }

        let (mut params, _) = self.parse_paren_comma_seq(|p| {
            p.recover_vcs_conflict_marker();
            let snapshot = p.create_snapshot_for_diagnostic();
            let param = p.parse_param_general(fn_parse_mode, first_param, true).or_else(|e| {
                let guar = e.emit();
                // When parsing a param failed, we should check to make the span of the param
                // not contain '(' before it.
                // For example when parsing `*mut Self` in function `fn oof(*mut Self)`.
                let lo = if let TokenKind::OpenParen = p.prev_token.kind {
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
    /// - `recover_arg_parse` is used to recover from a failed argument parse.
    pub(super) fn parse_param_general(
        &mut self,
        fn_parse_mode: &FnParseMode,
        first_param: bool,
        recover_arg_parse: bool,
    ) -> PResult<'a, Param> {
        let lo = self.token.span;
        let attrs = self.parse_outer_attributes()?;
        self.collect_tokens(None, attrs, ForceCollect::No, |this, attrs| {
            // Possibly parse `self`. Recover if we parsed it and it wasn't allowed here.
            if let Some(mut param) = this.parse_self_param()? {
                param.attrs = attrs;
                let res = if first_param { Ok(param) } else { this.recover_bad_self_param(param) };
                return Ok((res?, Trailing::No, UsePreAttrPos::No));
            }

            let is_dot_dot_dot = if this.token.kind == token::DotDotDot {
                IsDotDotDot::Yes
            } else {
                IsDotDotDot::No
            };
            let is_name_required = (fn_parse_mode.req_name)(
                this.token.span.with_neighbor(this.prev_token.span).edition(),
                is_dot_dot_dot,
            );
            let is_name_required = if is_name_required && is_dot_dot_dot == IsDotDotDot::Yes {
                this.psess.buffer_lint(
                    VARARGS_WITHOUT_PATTERN,
                    this.token.span,
                    ast::CRATE_NODE_ID,
                    diagnostics::VarargsWithoutPattern { span: this.token.span },
                );
                false
            } else {
                is_name_required
            };
            let (pat, ty) = if is_name_required || this.is_named_param() {
                debug!("parse_param_general parse_pat (is_name_required:{})", is_name_required);
                let (pat, colon) = this.parse_fn_param_pat_colon()?;
                if !colon {
                    let mut err = this.unexpected().unwrap_err();
                    let pat_span = pat.span;
                    return if let Some(ident) = this.parameter_without_type(
                        &mut err,
                        pat,
                        is_name_required,
                        first_param,
                        fn_parse_mode,
                    ) {
                        let guar = err.emit();
                        let mut arg = dummy_arg(ident, guar);
                        arg.span = pat_span;
                        Ok((arg, Trailing::No, UsePreAttrPos::No))
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
                    if let TyKind::Path(_, Path { segments, .. }) = &t.kind
                        && let Some(segment) = segments.last()
                        && let Some(guar) =
                            this.check_trailing_angle_brackets(segment, &[exp!(CloseParen)])
                    {
                        return Ok((
                            dummy_arg(segment.ident, guar),
                            Trailing::No,
                            UsePreAttrPos::No,
                        ));
                    }

                    if this.token != token::Comma && this.token != token::CloseParen {
                        // This wasn't actually a type, but a pattern looking like a type,
                        // so we are going to rollback and re-parse for recovery.
                        ty = this.unexpected_any();
                    }
                }
                match ty {
                    Ok(ty) => {
                        let pat = this.mk_pat(ty.span, PatKind::Missing);
                        (Box::new(pat), ty)
                    }
                    // If this is a C-variadic argument and we hit an error, return the error.
                    Err(err) if this.token == token::DotDotDot => return Err(err),
                    Err(err) if this.unmatched_angle_bracket_count > 0 => return Err(err),
                    Err(err) if recover_arg_parse => {
                        // Recover from attempting to parse the argument as a type without pattern.
                        err.cancel();
                        this.restore_snapshot(parser_snapshot_before_ty);
                        this.recover_arg_parse()?
                    }
                    Err(err) => return Err(err),
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
                            this.dcx().emit_err(diagnostics::IncorrectTypeOnSelf {
                                span: ty.span,
                                move_self_modifier: diagnostics::MoveSelfModifier {
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
            this.dcx().emit_err(diagnostics::SelfArgumentPointer { span: this.token.span });

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

        let eself = respan(eself_lo.to(eself_hi), eself);
        Ok(Some(Param::from_self(AttrVec::default(), eself, eself_ident)))
    }

    fn is_named_param(&self) -> bool {
        let offset = match &self.token.kind {
            token::OpenInvisible(origin) => match origin {
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

    pub(crate) fn recover_self_param(&mut self) -> bool {
        matches!(
            self.parse_outer_attributes()
                .and_then(|_| self.parse_self_param())
                .map_err(|e| e.cancel()),
            Ok(Some(_))
        )
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum FrontMatterParsingMode {
    /// Parse the front matter of a function declaration
    Function,
    /// Parse the front matter of a function pointer type.
    /// For function pointer types, the `const` and `async` keywords are not permitted.
    FunctionPtrType,
}
