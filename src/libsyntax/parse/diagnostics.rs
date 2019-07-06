use crate::ast::{
    self, Arg, BinOpKind, BindingMode, BlockCheckMode, Expr, ExprKind, Ident, Item, ItemKind,
    Mutability, Pat, PatKind, PathSegment, QSelf, Ty, TyKind, VariantData,
};
use crate::parse::{SeqSep, PResult, Parser, ParseSess};
use crate::parse::parser::{BlockMode, PathStyle, SemiColonMode, TokenType, TokenExpectType};
use crate::parse::token::{self, TokenKind};
use crate::print::pprust;
use crate::ptr::P;
use crate::source_map::Spanned;
use crate::symbol::{kw, sym};
use crate::ThinVec;
use crate::util::parser::AssocOp;
use errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_data_structures::fx::FxHashSet;
use syntax_pos::{Span, DUMMY_SP, MultiSpan};
use log::{debug, trace};

/// Creates a placeholder argument.
crate fn dummy_arg(ident: Ident) -> Arg {
    let pat = P(Pat {
        id: ast::DUMMY_NODE_ID,
        node: PatKind::Ident(BindingMode::ByValue(Mutability::Immutable), ident, None),
        span: ident.span,
    });
    let ty = Ty {
        node: TyKind::Err,
        span: ident.span,
        id: ast::DUMMY_NODE_ID
    };
    Arg { attrs: ThinVec::default(), id: ast::DUMMY_NODE_ID, pat, ty: P(ty) }
}

pub enum Error {
    FileNotFoundForModule {
        mod_name: String,
        default_path: String,
        secondary_path: String,
        dir_path: String,
    },
    DuplicatePaths {
        mod_name: String,
        default_path: String,
        secondary_path: String,
    },
    UselessDocComment,
    InclusiveRangeWithNoEnd,
}

impl Error {
    fn span_err<S: Into<MultiSpan>>(
        self,
        sp: S,
        handler: &errors::Handler,
    ) -> DiagnosticBuilder<'_> {
        match self {
            Error::FileNotFoundForModule {
                ref mod_name,
                ref default_path,
                ref secondary_path,
                ref dir_path,
            } => {
                let mut err = struct_span_err!(
                    handler,
                    sp,
                    E0583,
                    "file not found for module `{}`",
                    mod_name,
                );
                err.help(&format!(
                    "name the file either {} or {} inside the directory \"{}\"",
                    default_path,
                    secondary_path,
                    dir_path,
                ));
                err
            }
            Error::DuplicatePaths { ref mod_name, ref default_path, ref secondary_path } => {
                let mut err = struct_span_err!(
                    handler,
                    sp,
                    E0584,
                    "file for module `{}` found at both {} and {}",
                    mod_name,
                    default_path,
                    secondary_path,
                );
                err.help("delete or rename one of them to remove the ambiguity");
                err
            }
            Error::UselessDocComment => {
                let mut err = struct_span_err!(
                    handler,
                    sp,
                    E0585,
                    "found a documentation comment that doesn't document anything",
                );
                err.help("doc comments must come before what they document, maybe a comment was \
                          intended with `//`?");
                err
            }
            Error::InclusiveRangeWithNoEnd => {
                let mut err = struct_span_err!(
                    handler,
                    sp,
                    E0586,
                    "inclusive range with no end",
                );
                err.help("inclusive ranges must be bounded at the end (`..=b` or `a..=b`)");
                err
            }
        }
    }
}

pub trait RecoverQPath: Sized + 'static {
    const PATH_STYLE: PathStyle = PathStyle::Expr;
    fn to_ty(&self) -> Option<P<Ty>>;
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self;
}

impl RecoverQPath for Ty {
    const PATH_STYLE: PathStyle = PathStyle::Type;
    fn to_ty(&self) -> Option<P<Ty>> {
        Some(P(self.clone()))
    }
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            node: TyKind::Path(qself, path),
            id: ast::DUMMY_NODE_ID,
        }
    }
}

impl RecoverQPath for Pat {
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            node: PatKind::Path(qself, path),
            id: ast::DUMMY_NODE_ID,
        }
    }
}

impl RecoverQPath for Expr {
    fn to_ty(&self) -> Option<P<Ty>> {
        self.to_ty()
    }
    fn recovered(qself: Option<QSelf>, path: ast::Path) -> Self {
        Self {
            span: path.span,
            node: ExprKind::Path(qself, path),
            attrs: ThinVec::new(),
            id: ast::DUMMY_NODE_ID,
        }
    }
}

impl<'a> Parser<'a> {
    pub fn fatal(&self, m: &str) -> DiagnosticBuilder<'a> {
        self.span_fatal(self.token.span, m)
    }

    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_fatal(sp, m)
    }

    pub fn span_fatal_err<S: Into<MultiSpan>>(&self, sp: S, err: Error) -> DiagnosticBuilder<'a> {
        err.span_err(sp, self.diagnostic())
    }

    pub fn bug(&self, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(self.token.span, m)
    }

    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, m: &str) {
        self.sess.span_diagnostic.span_err(sp, m)
    }

    crate fn struct_span_err<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> DiagnosticBuilder<'a> {
        self.sess.span_diagnostic.struct_span_err(sp, m)
    }

    crate fn span_bug<S: Into<MultiSpan>>(&self, sp: S, m: &str) -> ! {
        self.sess.span_diagnostic.span_bug(sp, m)
    }

    crate fn cancel(&self, err: &mut DiagnosticBuilder<'_>) {
        self.sess.span_diagnostic.cancel(err)
    }

    crate fn diagnostic(&self) -> &'a errors::Handler {
        &self.sess.span_diagnostic
    }

    crate fn expected_ident_found(&self) -> DiagnosticBuilder<'a> {
        let mut err = self.struct_span_err(
            self.token.span,
            &format!("expected identifier, found {}", self.this_token_descr()),
        );
        if let token::Ident(name, false) = self.token.kind {
            if Ident::new(name, self.token.span).is_raw_guess() {
                err.span_suggestion(
                    self.token.span,
                    "you can escape reserved keywords to use them as identifiers",
                    format!("r#{}", name),
                    Applicability::MaybeIncorrect,
                );
            }
        }
        if let Some(token_descr) = self.token_descr() {
            err.span_label(self.token.span, format!("expected identifier, found {}", token_descr));
        } else {
            err.span_label(self.token.span, "expected identifier");
            if self.token == token::Comma && self.look_ahead(1, |t| t.is_ident()) {
                err.span_suggestion(
                    self.token.span,
                    "remove this comma",
                    String::new(),
                    Applicability::MachineApplicable,
                );
            }
        }
        err
    }

    pub fn expected_one_of_not_found(
        &mut self,
        edible: &[TokenKind],
        inedible: &[TokenKind],
    ) -> PResult<'a, bool /* recovered */> {
        fn tokens_to_string(tokens: &[TokenType]) -> String {
            let mut i = tokens.iter();
            // This might be a sign we need a connect method on Iterator.
            let b = i.next()
                     .map_or(String::new(), |t| t.to_string());
            i.enumerate().fold(b, |mut b, (i, a)| {
                if tokens.len() > 2 && i == tokens.len() - 2 {
                    b.push_str(", or ");
                } else if tokens.len() == 2 && i == tokens.len() - 2 {
                    b.push_str(" or ");
                } else {
                    b.push_str(", ");
                }
                b.push_str(&a.to_string());
                b
            })
        }

        let mut expected = edible.iter()
            .map(|x| TokenType::Token(x.clone()))
            .chain(inedible.iter().map(|x| TokenType::Token(x.clone())))
            .chain(self.expected_tokens.iter().cloned())
            .collect::<Vec<_>>();
        expected.sort_by_cached_key(|x| x.to_string());
        expected.dedup();
        let expect = tokens_to_string(&expected[..]);
        let actual = self.this_token_to_string();
        let (msg_exp, (label_sp, label_exp)) = if expected.len() > 1 {
            let short_expect = if expected.len() > 6 {
                format!("{} possible tokens", expected.len())
            } else {
                expect.clone()
            };
            (format!("expected one of {}, found `{}`", expect, actual),
                (self.sess.source_map().next_point(self.prev_span),
                format!("expected one of {} here", short_expect)))
        } else if expected.is_empty() {
            (format!("unexpected token: `{}`", actual),
                (self.prev_span, "unexpected token after this".to_string()))
        } else {
            (format!("expected {}, found `{}`", expect, actual),
                (self.sess.source_map().next_point(self.prev_span),
                format!("expected {} here", expect)))
        };
        self.last_unexpected_token_span = Some(self.token.span);
        let mut err = self.fatal(&msg_exp);
        if self.token.is_ident_named(sym::and) {
            err.span_suggestion_short(
                self.token.span,
                "use `&&` instead of `and` for the boolean operator",
                "&&".to_string(),
                Applicability::MaybeIncorrect,
            );
        }
        if self.token.is_ident_named(sym::or) {
            err.span_suggestion_short(
                self.token.span,
                "use `||` instead of `or` for the boolean operator",
                "||".to_string(),
                Applicability::MaybeIncorrect,
            );
        }
        let sp = if self.token == token::Eof {
            // This is EOF, don't want to point at the following char, but rather the last token
            self.prev_span
        } else {
            label_sp
        };
        match self.recover_closing_delimiter(&expected.iter().filter_map(|tt| match tt {
            TokenType::Token(t) => Some(t.clone()),
            _ => None,
        }).collect::<Vec<_>>(), err) {
            Err(e) => err = e,
            Ok(recovered) => {
                return Ok(recovered);
            }
        }

        let is_semi_suggestable = expected.iter().any(|t| match t {
            TokenType::Token(token::Semi) => true, // we expect a `;` here
            _ => false,
        }) && ( // a `;` would be expected before the current keyword
            self.token.is_keyword(kw::Break) ||
            self.token.is_keyword(kw::Continue) ||
            self.token.is_keyword(kw::For) ||
            self.token.is_keyword(kw::If) ||
            self.token.is_keyword(kw::Let) ||
            self.token.is_keyword(kw::Loop) ||
            self.token.is_keyword(kw::Match) ||
            self.token.is_keyword(kw::Return) ||
            self.token.is_keyword(kw::While)
        );
        let cm = self.sess.source_map();
        match (cm.lookup_line(self.token.span.lo()), cm.lookup_line(sp.lo())) {
            (Ok(ref a), Ok(ref b)) if a.line != b.line && is_semi_suggestable => {
                // The spans are in different lines, expected `;` and found `let` or `return`.
                // High likelihood that it is only a missing `;`.
                err.span_suggestion_short(
                    label_sp,
                    "a semicolon may be missing here",
                    ";".to_string(),
                    Applicability::MaybeIncorrect,
                );
                err.emit();
                return Ok(true);
            }
            (Ok(ref a), Ok(ref b)) if a.line == b.line => {
                // When the spans are in the same line, it means that the only content between
                // them is whitespace, point at the found token in that case:
                //
                // X |     () => { syntax error };
                //   |                    ^^^^^ expected one of 8 possible tokens here
                //
                // instead of having:
                //
                // X |     () => { syntax error };
                //   |                   -^^^^^ unexpected token
                //   |                   |
                //   |                   expected one of 8 possible tokens here
                err.span_label(self.token.span, label_exp);
            }
            _ if self.prev_span == syntax_pos::DUMMY_SP => {
                // Account for macro context where the previous span might not be
                // available to avoid incorrect output (#54841).
                err.span_label(self.token.span, "unexpected token");
            }
            _ => {
                err.span_label(sp, label_exp);
                err.span_label(self.token.span, "unexpected token");
            }
        }
        Err(err)
    }

    /// Eats and discards tokens until one of `kets` is encountered. Respects token trees,
    /// passes through any errors encountered. Used for error recovery.
    crate fn eat_to_tokens(&mut self, kets: &[&TokenKind]) {
        let handler = self.diagnostic();

        if let Err(ref mut err) = self.parse_seq_to_before_tokens(
            kets,
            SeqSep::none(),
            TokenExpectType::Expect,
            |p| Ok(p.parse_token_tree()),
        ) {
            handler.cancel(err);
        }
    }

    /// This function checks if there are trailing angle brackets and produces
    /// a diagnostic to suggest removing them.
    ///
    /// ```ignore (diagnostic)
    /// let _ = vec![1, 2, 3].into_iter().collect::<Vec<usize>>>>();
    ///                                                        ^^ help: remove extra angle brackets
    /// ```
    crate fn check_trailing_angle_brackets(&mut self, segment: &PathSegment, end: TokenKind) {
        // This function is intended to be invoked after parsing a path segment where there are two
        // cases:
        //
        // 1. A specific token is expected after the path segment.
        //    eg. `x.foo(`, `x.foo::<u32>(` (parenthesis - method call),
        //        `Foo::`, or `Foo::<Bar>::` (mod sep - continued path).
        // 2. No specific token is expected after the path segment.
        //    eg. `x.foo` (field access)
        //
        // This function is called after parsing `.foo` and before parsing the token `end` (if
        // present). This includes any angle bracket arguments, such as `.foo::<u32>` or
        // `Foo::<Bar>`.

        // We only care about trailing angle brackets if we previously parsed angle bracket
        // arguments. This helps stop us incorrectly suggesting that extra angle brackets be
        // removed in this case:
        //
        // `x.foo >> (3)` (where `x.foo` is a `u32` for example)
        //
        // This case is particularly tricky as we won't notice it just looking at the tokens -
        // it will appear the same (in terms of upcoming tokens) as below (since the `::<u32>` will
        // have already been parsed):
        //
        // `x.foo::<u32>>>(3)`
        let parsed_angle_bracket_args = segment.args
            .as_ref()
            .map(|args| args.is_angle_bracketed())
            .unwrap_or(false);

        debug!(
            "check_trailing_angle_brackets: parsed_angle_bracket_args={:?}",
            parsed_angle_bracket_args,
        );
        if !parsed_angle_bracket_args {
            return;
        }

        // Keep the span at the start so we can highlight the sequence of `>` characters to be
        // removed.
        let lo = self.token.span;

        // We need to look-ahead to see if we have `>` characters without moving the cursor forward
        // (since we might have the field access case and the characters we're eating are
        // actual operators and not trailing characters - ie `x.foo >> 3`).
        let mut position = 0;

        // We can encounter `>` or `>>` tokens in any order, so we need to keep track of how
        // many of each (so we can correctly pluralize our error messages) and continue to
        // advance.
        let mut number_of_shr = 0;
        let mut number_of_gt = 0;
        while self.look_ahead(position, |t| {
            trace!("check_trailing_angle_brackets: t={:?}", t);
            if *t == token::BinOp(token::BinOpToken::Shr) {
                number_of_shr += 1;
                true
            } else if *t == token::Gt {
                number_of_gt += 1;
                true
            } else {
                false
            }
        }) {
            position += 1;
        }

        // If we didn't find any trailing `>` characters, then we have nothing to error about.
        debug!(
            "check_trailing_angle_brackets: number_of_gt={:?} number_of_shr={:?}",
            number_of_gt, number_of_shr,
        );
        if number_of_gt < 1 && number_of_shr < 1 {
            return;
        }

        // Finally, double check that we have our end token as otherwise this is the
        // second case.
        if self.look_ahead(position, |t| {
            trace!("check_trailing_angle_brackets: t={:?}", t);
            *t == end
        }) {
            // Eat from where we started until the end token so that parsing can continue
            // as if we didn't have those extra angle brackets.
            self.eat_to_tokens(&[&end]);
            let span = lo.until(self.token.span);

            let plural = number_of_gt > 1 || number_of_shr >= 1;
            self.diagnostic()
                .struct_span_err(
                    span,
                    &format!("unmatched angle bracket{}", if plural { "s" } else { "" }),
                )
                .span_suggestion(
                    span,
                    &format!("remove extra angle bracket{}", if plural { "s" } else { "" }),
                    String::new(),
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }

    /// Produce an error if comparison operators are chained (RFC #558).
    /// We only need to check lhs, not rhs, because all comparison ops
    /// have same precedence and are left-associative
    crate fn check_no_chained_comparison(&self, lhs: &Expr, outer_op: &AssocOp) {
        debug_assert!(outer_op.is_comparison(),
                      "check_no_chained_comparison: {:?} is not comparison",
                      outer_op);
        match lhs.node {
            ExprKind::Binary(op, _, _) if op.node.is_comparison() => {
                // respan to include both operators
                let op_span = op.span.to(self.token.span);
                let mut err = self.diagnostic().struct_span_err(op_span,
                    "chained comparison operators require parentheses");
                if op.node == BinOpKind::Lt &&
                    *outer_op == AssocOp::Less ||  // Include `<` to provide this recommendation
                    *outer_op == AssocOp::Greater  // even in a case like the following:
                {                                  //     Foo<Bar<Baz<Qux, ()>>>
                    err.help(
                        "use `::<...>` instead of `<...>` if you meant to specify type arguments");
                    err.help("or use `(...)` if you meant to specify fn arguments");
                }
                err.emit();
            }
            _ => {}
        }
    }

    crate fn maybe_report_ambiguous_plus(
        &mut self,
        allow_plus: bool,
        impl_dyn_multi: bool,
        ty: &Ty,
    ) {
        if !allow_plus && impl_dyn_multi {
            let sum_with_parens = format!("({})", pprust::ty_to_string(&ty));
            self.struct_span_err(ty.span, "ambiguous `+` in a type")
                .span_suggestion(
                    ty.span,
                    "use parentheses to disambiguate",
                    sum_with_parens,
                    Applicability::MachineApplicable,
                )
                .emit();
        }
    }

    crate fn maybe_report_invalid_custom_discriminants(
        sess: &ParseSess,
        variants: &[Spanned<ast::Variant_>],
    ) {
        let has_fields = variants.iter().any(|variant| match variant.node.data {
            VariantData::Tuple(..) | VariantData::Struct(..) => true,
            VariantData::Unit(..) => false,
        });

        let discriminant_spans = variants.iter().filter(|variant| match variant.node.data {
            VariantData::Tuple(..) | VariantData::Struct(..) => false,
            VariantData::Unit(..) => true,
        })
        .filter_map(|variant| variant.node.disr_expr.as_ref().map(|c| c.value.span))
        .collect::<Vec<_>>();

        if !discriminant_spans.is_empty() && has_fields {
            let mut err = crate::feature_gate::feature_err(
                sess,
                sym::arbitrary_enum_discriminant,
                discriminant_spans.clone(),
                crate::feature_gate::GateIssue::Language,
                "custom discriminant values are not allowed in enums with tuple or struct variants",
            );
            for sp in discriminant_spans {
                err.span_label(sp, "disallowed custom discriminant");
            }
            for variant in variants.iter() {
                match &variant.node.data {
                    VariantData::Struct(..) => {
                        err.span_label(
                            variant.span,
                            "struct variant defined here",
                        );
                    }
                    VariantData::Tuple(..) => {
                        err.span_label(
                            variant.span,
                            "tuple variant defined here",
                        );
                    }
                    VariantData::Unit(..) => {}
                }
            }
            err.emit();
        }
    }

    crate fn maybe_recover_from_bad_type_plus(
        &mut self,
        allow_plus: bool,
        ty: &Ty,
    ) -> PResult<'a, ()> {
        // Do not add `+` to expected tokens.
        if !allow_plus || !self.token.is_like_plus() {
            return Ok(());
        }

        self.bump(); // `+`
        let bounds = self.parse_generic_bounds(None)?;
        let sum_span = ty.span.to(self.prev_span);

        let mut err = struct_span_err!(
            self.sess.span_diagnostic,
            sum_span,
            E0178,
            "expected a path on the left-hand side of `+`, not `{}`",
            pprust::ty_to_string(ty)
        );

        match ty.node {
            TyKind::Rptr(ref lifetime, ref mut_ty) => {
                let sum_with_parens = pprust::to_string(|s| {
                    use crate::print::pprust::PrintState;

                    s.s.word("&");
                    s.print_opt_lifetime(lifetime);
                    s.print_mutability(mut_ty.mutbl);
                    s.popen();
                    s.print_type(&mut_ty.ty);
                    s.print_type_bounds(" +", &bounds);
                    s.pclose()
                });
                err.span_suggestion(
                    sum_span,
                    "try adding parentheses",
                    sum_with_parens,
                    Applicability::MachineApplicable,
                );
            }
            TyKind::Ptr(..) | TyKind::BareFn(..) => {
                err.span_label(sum_span, "perhaps you forgot parentheses?");
            }
            _ => {
                err.span_label(sum_span, "expected a path");
            }
        }
        err.emit();
        Ok(())
    }

    /// Try to recover from associated item paths like `[T]::AssocItem`/`(T, U)::AssocItem`.
    /// Attempt to convert the base expression/pattern/type into a type, parse the `::AssocItem`
    /// tail, and combine them into a `<Ty>::AssocItem` expression/pattern/type.
    crate fn maybe_recover_from_bad_qpath<T: RecoverQPath>(
        &mut self,
        base: P<T>,
        allow_recovery: bool,
    ) -> PResult<'a, P<T>> {
        // Do not add `::` to expected tokens.
        if allow_recovery && self.token == token::ModSep {
            if let Some(ty) = base.to_ty() {
                return self.maybe_recover_from_bad_qpath_stage_2(ty.span, ty);
            }
        }
        Ok(base)
    }

    /// Given an already parsed `Ty` parse the `::AssocItem` tail and
    /// combine them into a `<Ty>::AssocItem` expression/pattern/type.
    crate fn maybe_recover_from_bad_qpath_stage_2<T: RecoverQPath>(
        &mut self,
        ty_span: Span,
        ty: P<Ty>,
    ) -> PResult<'a, P<T>> {
        self.expect(&token::ModSep)?;

        let mut path = ast::Path {
            segments: Vec::new(),
            span: DUMMY_SP,
        };
        self.parse_path_segments(&mut path.segments, T::PATH_STYLE)?;
        path.span = ty_span.to(self.prev_span);

        let ty_str = self
            .sess
            .source_map()
            .span_to_snippet(ty_span)
            .unwrap_or_else(|_| pprust::ty_to_string(&ty));
        self.diagnostic()
            .struct_span_err(path.span, "missing angle brackets in associated item path")
            .span_suggestion(
                // this is a best-effort recovery
                path.span,
                "try",
                format!("<{}>::{}", ty_str, path),
                Applicability::MaybeIncorrect,
            )
            .emit();

        let path_span = ty_span.shrink_to_hi(); // use an empty path since `position` == 0
        Ok(P(T::recovered(
            Some(QSelf {
                ty,
                path_span,
                position: 0,
            }),
            path,
        )))
    }

    crate fn maybe_consume_incorrect_semicolon(&mut self, items: &[P<Item>]) -> bool {
        if self.eat(&token::Semi) {
            let mut err = self.struct_span_err(self.prev_span, "expected item, found `;`");
            err.span_suggestion_short(
                self.prev_span,
                "remove this semicolon",
                String::new(),
                Applicability::MachineApplicable,
            );
            if !items.is_empty() {
                let previous_item = &items[items.len() - 1];
                let previous_item_kind_name = match previous_item.node {
                    // say "braced struct" because tuple-structs and
                    // braceless-empty-struct declarations do take a semicolon
                    ItemKind::Struct(..) => Some("braced struct"),
                    ItemKind::Enum(..) => Some("enum"),
                    ItemKind::Trait(..) => Some("trait"),
                    ItemKind::Union(..) => Some("union"),
                    _ => None,
                };
                if let Some(name) = previous_item_kind_name {
                    err.help(&format!(
                        "{} declarations are not followed by a semicolon",
                        name
                    ));
                }
            }
            err.emit();
            true
        } else {
            false
        }
    }

    /// Create a `DiagnosticBuilder` for an unexpected token `t` and try to recover if it is a
    /// closing delimiter.
    pub fn unexpected_try_recover(
        &mut self,
        t: &TokenKind,
    ) -> PResult<'a, bool /* recovered */> {
        let token_str = pprust::token_kind_to_string(t);
        let this_token_str = self.this_token_descr();
        let (prev_sp, sp) = match (&self.token.kind, self.subparser_name) {
            // Point at the end of the macro call when reaching end of macro arguments.
            (token::Eof, Some(_)) => {
                let sp = self.sess.source_map().next_point(self.token.span);
                (sp, sp)
            }
            // We don't want to point at the following span after DUMMY_SP.
            // This happens when the parser finds an empty TokenStream.
            _ if self.prev_span == DUMMY_SP => (self.token.span, self.token.span),
            // EOF, don't want to point at the following char, but rather the last token.
            (token::Eof, None) => (self.prev_span, self.token.span),
            _ => (self.sess.source_map().next_point(self.prev_span), self.token.span),
        };
        let msg = format!(
            "expected `{}`, found {}",
            token_str,
            match (&self.token.kind, self.subparser_name) {
                (token::Eof, Some(origin)) => format!("end of {}", origin),
                _ => this_token_str,
            },
        );
        let mut err = self.struct_span_err(sp, &msg);
        let label_exp = format!("expected `{}`", token_str);
        match self.recover_closing_delimiter(&[t.clone()], err) {
            Err(e) => err = e,
            Ok(recovered) => {
                return Ok(recovered);
            }
        }
        let cm = self.sess.source_map();
        match (cm.lookup_line(prev_sp.lo()), cm.lookup_line(sp.lo())) {
            (Ok(ref a), Ok(ref b)) if a.line == b.line => {
                // When the spans are in the same line, it means that the only content
                // between them is whitespace, point only at the found token.
                err.span_label(sp, label_exp);
            }
            _ => {
                err.span_label(prev_sp, label_exp);
                err.span_label(sp, "unexpected token");
            }
        }
        Err(err)
    }

    /// Consume alternative await syntaxes like `await <expr>`, `await? <expr>`, `await(<expr>)`
    /// and `await { <expr> }`.
    crate fn parse_incorrect_await_syntax(
        &mut self,
        lo: Span,
        await_sp: Span,
    ) -> PResult<'a, (Span, ExprKind)> {
        let is_question = self.eat(&token::Question); // Handle `await? <expr>`.
        let expr = if self.token == token::OpenDelim(token::Brace) {
            // Handle `await { <expr> }`.
            // This needs to be handled separatedly from the next arm to avoid
            // interpreting `await { <expr> }?` as `<expr>?.await`.
            self.parse_block_expr(
                None,
                self.token.span,
                BlockCheckMode::Default,
                ThinVec::new(),
            )
        } else {
            self.parse_expr()
        }.map_err(|mut err| {
            err.span_label(await_sp, "while parsing this incorrect await expression");
            err
        })?;
        let expr_str = self.sess.source_map().span_to_snippet(expr.span)
            .unwrap_or_else(|_| pprust::expr_to_string(&expr));
        let suggestion = format!("{}.await{}", expr_str, if is_question { "?" } else { "" });
        let sp = lo.to(expr.span);
        let app = match expr.node {
            ExprKind::Try(_) => Applicability::MaybeIncorrect, // `await <expr>?`
            _ => Applicability::MachineApplicable,
        };
        self.struct_span_err(sp, "incorrect use of `await`")
            .span_suggestion(sp, "`await` is a postfix operation", suggestion, app)
            .emit();
        Ok((sp, ExprKind::Await(ast::AwaitOrigin::FieldLike, expr)))
    }

    /// If encountering `future.await()`, consume and emit error.
    crate fn recover_from_await_method_call(&mut self) {
        if self.token == token::OpenDelim(token::Paren) &&
            self.look_ahead(1, |t| t == &token::CloseDelim(token::Paren))
        {
            // future.await()
            let lo = self.token.span;
            self.bump(); // (
            let sp = lo.to(self.token.span);
            self.bump(); // )
            self.struct_span_err(sp, "incorrect use of `await`")
                .span_suggestion(
                    sp,
                    "`await` is not a method call, remove the parentheses",
                    String::new(),
                    Applicability::MachineApplicable,
                ).emit()
        }
    }

    crate fn could_ascription_be_path(&self, node: &ast::ExprKind) -> bool {
        self.token.is_ident() &&
            if let ast::ExprKind::Path(..) = node { true } else { false } &&
            !self.token.is_reserved_ident() &&           // v `foo:bar(baz)`
            self.look_ahead(1, |t| t == &token::OpenDelim(token::Paren)) ||
            self.look_ahead(1, |t| t == &token::Lt) &&     // `foo:bar<baz`
            self.look_ahead(2, |t| t.is_ident()) ||
            self.look_ahead(1, |t| t == &token::Colon) &&  // `foo:bar:baz`
            self.look_ahead(2, |t| t.is_ident()) ||
            self.look_ahead(1, |t| t == &token::ModSep) &&  // `foo:bar::baz`
            self.look_ahead(2, |t| t.is_ident())
    }

    crate fn bad_type_ascription(
        &self,
        err: &mut DiagnosticBuilder<'a>,
        lhs_span: Span,
        cur_op_span: Span,
        next_sp: Span,
        maybe_path: bool,
    ) {
        err.span_label(self.token.span, "expecting a type here because of type ascription");
        let cm = self.sess.source_map();
        let next_pos = cm.lookup_char_pos(next_sp.lo());
        let op_pos = cm.lookup_char_pos(cur_op_span.hi());
        if op_pos.line != next_pos.line {
            err.span_suggestion(
                cur_op_span,
                "try using a semicolon",
                ";".to_string(),
                Applicability::MaybeIncorrect,
            );
        } else {
            if maybe_path {
                err.span_suggestion(
                    cur_op_span,
                    "maybe you meant to write a path separator here",
                    "::".to_string(),
                    Applicability::MaybeIncorrect,
                );
            } else {
                err.note("#![feature(type_ascription)] lets you annotate an \
                          expression with a type: `<expr>: <type>`")
                    .span_note(
                        lhs_span,
                        "this expression expects an ascribed type after the colon",
                    )
                    .help("this might be indicative of a syntax error elsewhere");
            }
        }
    }

    crate fn recover_seq_parse_error(
        &mut self,
        delim: token::DelimToken,
        lo: Span,
        result: PResult<'a, P<Expr>>,
    ) -> P<Expr> {
        match result {
            Ok(x) => x,
            Err(mut err) => {
                err.emit();
                // recover from parse error
                self.consume_block(delim);
                self.mk_expr(lo.to(self.prev_span), ExprKind::Err, ThinVec::new())
            }
        }
    }

    crate fn recover_closing_delimiter(
        &mut self,
        tokens: &[TokenKind],
        mut err: DiagnosticBuilder<'a>,
    ) -> PResult<'a, bool> {
        let mut pos = None;
        // we want to use the last closing delim that would apply
        for (i, unmatched) in self.unclosed_delims.iter().enumerate().rev() {
            if tokens.contains(&token::CloseDelim(unmatched.expected_delim))
                && Some(self.token.span) > unmatched.unclosed_span
            {
                pos = Some(i);
            }
        }
        match pos {
            Some(pos) => {
                // Recover and assume that the detected unclosed delimiter was meant for
                // this location. Emit the diagnostic and act as if the delimiter was
                // present for the parser's sake.

                 // Don't attempt to recover from this unclosed delimiter more than once.
                let unmatched = self.unclosed_delims.remove(pos);
                let delim = TokenType::Token(token::CloseDelim(unmatched.expected_delim));

                 // We want to suggest the inclusion of the closing delimiter where it makes
                // the most sense, which is immediately after the last token:
                //
                //  {foo(bar {}}
                //      -      ^
                //      |      |
                //      |      help: `)` may belong here
                //      |
                //      unclosed delimiter
                if let Some(sp) = unmatched.unclosed_span {
                    err.span_label(sp, "unclosed delimiter");
                }
                err.span_suggestion_short(
                    self.sess.source_map().next_point(self.prev_span),
                    &format!("{} may belong here", delim.to_string()),
                    delim.to_string(),
                    Applicability::MaybeIncorrect,
                );
                err.emit();
                self.expected_tokens.clear();  // reduce errors
                Ok(true)
            }
            _ => Err(err),
        }
    }

    /// Recover from `pub` keyword in places where it seems _reasonable_ but isn't valid.
    crate fn eat_bad_pub(&mut self) {
        if self.token.is_keyword(kw::Pub) {
            match self.parse_visibility(false) {
                Ok(vis) => {
                    self.diagnostic()
                        .struct_span_err(vis.span, "unnecessary visibility qualifier")
                        .span_label(vis.span, "`pub` not permitted here")
                        .emit();
                }
                Err(mut err) => err.emit(),
            }
        }
    }

    // Eat tokens until we can be relatively sure we reached the end of the
    // statement. This is something of a best-effort heuristic.
    //
    // We terminate when we find an unmatched `}` (without consuming it).
    crate fn recover_stmt(&mut self) {
        self.recover_stmt_(SemiColonMode::Ignore, BlockMode::Ignore)
    }

    // If `break_on_semi` is `Break`, then we will stop consuming tokens after
    // finding (and consuming) a `;` outside of `{}` or `[]` (note that this is
    // approximate - it can mean we break too early due to macros, but that
    // should only lead to sub-optimal recovery, not inaccurate parsing).
    //
    // If `break_on_block` is `Break`, then we will stop consuming tokens
    // after finding (and consuming) a brace-delimited block.
    crate fn recover_stmt_(&mut self, break_on_semi: SemiColonMode, break_on_block: BlockMode) {
        let mut brace_depth = 0;
        let mut bracket_depth = 0;
        let mut in_block = false;
        debug!("recover_stmt_ enter loop (semi={:?}, block={:?})",
               break_on_semi, break_on_block);
        loop {
            debug!("recover_stmt_ loop {:?}", self.token);
            match self.token.kind {
                token::OpenDelim(token::DelimToken::Brace) => {
                    brace_depth += 1;
                    self.bump();
                    if break_on_block == BlockMode::Break &&
                       brace_depth == 1 &&
                       bracket_depth == 0 {
                        in_block = true;
                    }
                }
                token::OpenDelim(token::DelimToken::Bracket) => {
                    bracket_depth += 1;
                    self.bump();
                }
                token::CloseDelim(token::DelimToken::Brace) => {
                    if brace_depth == 0 {
                        debug!("recover_stmt_ return - close delim {:?}", self.token);
                        break;
                    }
                    brace_depth -= 1;
                    self.bump();
                    if in_block && bracket_depth == 0 && brace_depth == 0 {
                        debug!("recover_stmt_ return - block end {:?}", self.token);
                        break;
                    }
                }
                token::CloseDelim(token::DelimToken::Bracket) => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        bracket_depth = 0;
                    }
                    self.bump();
                }
                token::Eof => {
                    debug!("recover_stmt_ return - Eof");
                    break;
                }
                token::Semi => {
                    self.bump();
                    if break_on_semi == SemiColonMode::Break &&
                       brace_depth == 0 &&
                       bracket_depth == 0 {
                        debug!("recover_stmt_ return - Semi");
                        break;
                    }
                }
                token::Comma if break_on_semi == SemiColonMode::Comma &&
                       brace_depth == 0 &&
                       bracket_depth == 0 =>
                {
                    debug!("recover_stmt_ return - Semi");
                    break;
                }
                _ => {
                    self.bump()
                }
            }
        }
    }

    crate fn check_for_for_in_in_typo(&mut self, in_span: Span) {
        if self.eat_keyword(kw::In) {
            // a common typo: `for _ in in bar {}`
            let mut err = self.sess.span_diagnostic.struct_span_err(
                self.prev_span,
                "expected iterable, found keyword `in`",
            );
            err.span_suggestion_short(
                in_span.until(self.prev_span),
                "remove the duplicated `in`",
                String::new(),
                Applicability::MachineApplicable,
            );
            err.emit();
        }
    }

    crate fn expected_semi_or_open_brace(&mut self) -> PResult<'a, ast::TraitItem> {
        let token_str = self.this_token_descr();
        let mut err = self.fatal(&format!("expected `;` or `{{`, found {}", token_str));
        err.span_label(self.token.span, "expected `;` or `{`");
        Err(err)
    }

    crate fn eat_incorrect_doc_comment_for_arg_type(&mut self) {
        if let token::DocComment(_) = self.token.kind {
            let mut err = self.diagnostic().struct_span_err(
                self.token.span,
                "documentation comments cannot be applied to a function parameter's type",
            );
            err.span_label(self.token.span, "doc comments are not allowed here");
            err.emit();
            self.bump();
        } else if self.token == token::Pound && self.look_ahead(1, |t| {
            *t == token::OpenDelim(token::Bracket)
        }) {
            let lo = self.token.span;
            // Skip every token until next possible arg.
            while self.token != token::CloseDelim(token::Bracket) {
                self.bump();
            }
            let sp = lo.to(self.token.span);
            self.bump();
            let mut err = self.diagnostic().struct_span_err(
                sp,
                "attributes cannot be applied to a function parameter's type",
            );
            err.span_label(sp, "attributes are not allowed here");
            err.emit();
        }
    }

    crate fn argument_without_type(
        &mut self,
        err: &mut DiagnosticBuilder<'_>,
        pat: P<ast::Pat>,
        require_name: bool,
        is_trait_item: bool,
    ) -> Option<Ident> {
        // If we find a pattern followed by an identifier, it could be an (incorrect)
        // C-style parameter declaration.
        if self.check_ident() && self.look_ahead(1, |t| {
            *t == token::Comma || *t == token::CloseDelim(token::Paren)
        }) { // `fn foo(String s) {}`
            let ident = self.parse_ident().unwrap();
            let span = pat.span.with_hi(ident.span.hi());

            err.span_suggestion(
                span,
                "declare the type after the parameter binding",
                String::from("<identifier>: <type>"),
                Applicability::HasPlaceholders,
            );
            return Some(ident);
        } else if let PatKind::Ident(_, ident, _) = pat.node {
            if require_name && (
                is_trait_item ||
                self.token == token::Comma ||
                self.token == token::CloseDelim(token::Paren)
            ) { // `fn foo(a, b) {}` or `fn foo(usize, usize) {}`
                err.span_suggestion(
                    pat.span,
                    "if this was a parameter name, give it a type",
                    format!("{}: TypeName", ident),
                    Applicability::HasPlaceholders,
                );
                err.span_suggestion(
                    pat.span,
                    "if this is a type, explicitly ignore the parameter name",
                    format!("_: {}", ident),
                    Applicability::MachineApplicable,
                );
                err.note("anonymous parameters are removed in the 2018 edition (see RFC 1685)");
                return Some(ident);
            }
        }
        None
    }

    crate fn recover_arg_parse(&mut self) -> PResult<'a, (P<ast::Pat>, P<ast::Ty>)> {
        let pat = self.parse_pat(Some("argument name"))?;
        self.expect(&token::Colon)?;
        let ty = self.parse_ty()?;

        let mut err = self.diagnostic().struct_span_err_with_code(
            pat.span,
            "patterns aren't allowed in methods without bodies",
            DiagnosticId::Error("E0642".into()),
        );
        err.span_suggestion_short(
            pat.span,
            "give this argument a name or use an underscore to ignore it",
            "_".to_owned(),
            Applicability::MachineApplicable,
        );
        err.emit();

        // Pretend the pattern is `_`, to avoid duplicate errors from AST validation.
        let pat = P(Pat {
            node: PatKind::Wild,
            span: pat.span,
            id: ast::DUMMY_NODE_ID
        });
        Ok((pat, ty))
    }

    crate fn recover_bad_self_arg(
        &mut self,
        mut arg: ast::Arg,
        is_trait_item: bool,
    ) -> PResult<'a, ast::Arg> {
        let sp = arg.pat.span;
        arg.ty.node = TyKind::Err;
        let mut err = self.struct_span_err(sp, "unexpected `self` parameter in function");
        if is_trait_item {
            err.span_label(sp, "must be the first associated function parameter");
        } else {
            err.span_label(sp, "not valid as function parameter");
            err.note("`self` is only valid as the first parameter of an associated function");
        }
        err.emit();
        Ok(arg)
    }

    crate fn consume_block(&mut self, delim: token::DelimToken) {
        let mut brace_depth = 0;
        loop {
            if self.eat(&token::OpenDelim(delim)) {
                brace_depth += 1;
            } else if self.eat(&token::CloseDelim(delim)) {
                if brace_depth == 0 {
                    return;
                } else {
                    brace_depth -= 1;
                    continue;
                }
            } else if self.token == token::Eof || self.eat(&token::CloseDelim(token::NoDelim)) {
                return;
            } else {
                self.bump();
            }
        }
    }

    crate fn expected_expression_found(&self) -> DiagnosticBuilder<'a> {
        let (span, msg) = match (&self.token.kind, self.subparser_name) {
            (&token::Eof, Some(origin)) => {
                let sp = self.sess.source_map().next_point(self.token.span);
                (sp, format!("expected expression, found end of {}", origin))
            }
            _ => (self.token.span, format!(
                "expected expression, found {}",
                self.this_token_descr(),
            )),
        };
        let mut err = self.struct_span_err(span, &msg);
        let sp = self.sess.source_map().start_point(self.token.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            self.sess.expr_parentheses_needed(&mut err, *sp, None);
        }
        err.span_label(span, "expected expression");
        err
    }

    /// Replace duplicated recovered arguments with `_` pattern to avoid unecessary errors.
    ///
    /// This is necessary because at this point we don't know whether we parsed a function with
    /// anonymous arguments or a function with names but no types. In order to minimize
    /// unecessary errors, we assume the arguments are in the shape of `fn foo(a, b, c)` where
    /// the arguments are *names* (so we don't emit errors about not being able to find `b` in
    /// the local scope), but if we find the same name multiple times, like in `fn foo(i8, i8)`,
    /// we deduplicate them to not complain about duplicated argument names.
    crate fn deduplicate_recovered_arg_names(&self, fn_inputs: &mut Vec<Arg>) {
        let mut seen_inputs = FxHashSet::default();
        for input in fn_inputs.iter_mut() {
            let opt_ident = if let (PatKind::Ident(_, ident, _), TyKind::Err) = (
                &input.pat.node, &input.ty.node,
            ) {
                Some(*ident)
            } else {
                None
            };
            if let Some(ident) = opt_ident {
                if seen_inputs.contains(&ident) {
                    input.pat.node = PatKind::Wild;
                }
                seen_inputs.insert(ident);
            }
        }
    }
}
