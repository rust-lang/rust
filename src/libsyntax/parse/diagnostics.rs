use crate::ast;
use crate::ast::{
    BlockCheckMode, Expr, ExprKind, Item, ItemKind, Pat, PatKind, QSelf, Ty, TyKind, VariantData,
};
use crate::parse::parser::{BlockMode, PathStyle, SemiColonMode, TokenType};
use crate::parse::token;
use crate::parse::PResult;
use crate::parse::Parser;
use crate::print::pprust;
use crate::ptr::P;
use crate::source_map::Spanned;
use crate::symbol::kw;
use crate::ThinVec;
use errors::{Applicability, DiagnosticBuilder};
use log::debug;
use syntax_pos::{Span, DUMMY_SP};

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
        &mut self,
        discriminant_spans: Vec<Span>,
        variants: &[Spanned<ast::Variant_>],
    ) {
        let has_fields = variants.iter().any(|variant| match variant.node.data {
            VariantData::Tuple(..) | VariantData::Struct(..) => true,
            VariantData::Unit(..) => false,
        });

        if !discriminant_spans.is_empty() && has_fields {
            let mut err = self.struct_span_err(
                discriminant_spans.clone(),
                "custom discriminant values are not allowed in enums with fields",
            );
            for sp in discriminant_spans {
                err.span_label(sp, "invalid custom discriminant");
            }
            for variant in variants.iter() {
                if let VariantData::Struct(fields, ..) | VariantData::Tuple(fields, ..) =
                    &variant.node.data
                {
                    let fields = if fields.len() > 1 {
                        "fields"
                    } else {
                        "a field"
                    };
                    err.span_label(
                        variant.span,
                        &format!("variant with {fields} defined here", fields = fields),
                    );

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

                    s.s.word("&")?;
                    s.print_opt_lifetime(lifetime)?;
                    s.print_mutability(mut_ty.mutbl)?;
                    s.popen()?;
                    s.print_type(&mut_ty.ty)?;
                    s.print_type_bounds(" +", &bounds)?;
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
        t: &token::Token,
    ) -> PResult<'a, bool /* recovered */> {
        let token_str = pprust::token_to_string(t);
        let this_token_str = self.this_token_descr();
        let (prev_sp, sp) = match (&self.token, self.subparser_name) {
            // Point at the end of the macro call when reaching end of macro arguments.
            (token::Token::Eof, Some(_)) => {
                let sp = self.sess.source_map().next_point(self.span);
                (sp, sp)
            }
            // We don't want to point at the following span after DUMMY_SP.
            // This happens when the parser finds an empty TokenStream.
            _ if self.prev_span == DUMMY_SP => (self.span, self.span),
            // EOF, don't want to point at the following char, but rather the last token.
            (token::Token::Eof, None) => (self.prev_span, self.span),
            _ => (self.sess.source_map().next_point(self.prev_span), self.span),
        };
        let msg = format!(
            "expected `{}`, found {}",
            token_str,
            match (&self.token, self.subparser_name) {
                (token::Token::Eof, Some(origin)) => format!("end of {}", origin),
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
                self.span,
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
            let lo = self.span;
            self.bump(); // (
            let sp = lo.to(self.span);
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
        err.span_label(self.span, "expecting a type here because of type ascription");
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
                err.note("type ascription is a nightly-only feature that lets \
                          you annotate an expression with a type: `<expr>: <type>`")
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
        tokens: &[token::Token],
        mut err: DiagnosticBuilder<'a>,
    ) -> PResult<'a, bool> {
        let mut pos = None;
        // we want to use the last closing delim that would apply
        for (i, unmatched) in self.unclosed_delims.iter().enumerate().rev() {
            if tokens.contains(&token::CloseDelim(unmatched.expected_delim))
                && Some(self.span) > unmatched.unclosed_span
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
                //      |      help: `)` may belong here (FIXME: #58270)
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
            match self.token {
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
        let (span, msg) = match (&self.token, self.subparser_name) {
            (&token::Token::Eof, Some(origin)) => {
                let sp = self.sess.source_map().next_point(self.span);
                (sp, format!("expected expression, found end of {}", origin))
            }
            _ => (self.span, format!(
                "expected expression, found {}",
                self.this_token_descr(),
            )),
        };
        let mut err = self.struct_span_err(span, &msg);
        let sp = self.sess.source_map().start_point(self.span);
        if let Some(sp) = self.sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            self.sess.expr_parentheses_needed(&mut err, *sp, None);
        }
        err.span_label(span, "expected expression");
        err
    }
}
