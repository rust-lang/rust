use super::ty::{AllowPlus, RecoverQPath, RecoverReturnSign};
use super::{Parser, Restrictions, TokenType};
use crate::maybe_whole;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, Delimiter, Token, TokenKind};
use rustc_ast::{
    self as ast, AngleBracketedArg, AngleBracketedArgs, AnonConst, AssocConstraint,
    AssocConstraintKind, BlockCheckMode, GenericArg, GenericArgs, Generics, ParenthesizedArgs,
    Path, PathSegment, QSelf,
};
use rustc_errors::{pluralize, Applicability, PResult};
use rustc_span::source_map::{BytePos, Span};
use rustc_span::symbol::{kw, sym, Ident};
use std::mem;
use thin_vec::ThinVec;
use tracing::debug;

/// Specifies how to parse a path.
#[derive(Copy, Clone, PartialEq)]
pub enum PathStyle {
    /// In some contexts, notably in expressions, paths with generic arguments are ambiguous
    /// with something else. For example, in expressions `segment < ....` can be interpreted
    /// as a comparison and `segment ( ....` can be interpreted as a function call.
    /// In all such contexts the non-path interpretation is preferred by default for practical
    /// reasons, but the path interpretation can be forced by the disambiguator `::`, e.g.
    /// `x<y>` - comparisons, `x::<y>` - unambiguously a path.
    Expr,
    /// In other contexts, notably in types, no ambiguity exists and paths can be written
    /// without the disambiguator, e.g., `x<y>` - unambiguously a path.
    /// Paths with disambiguators are still accepted, `x::<Y>` - unambiguously a path too.
    Type,
    /// A path with generic arguments disallowed, e.g., `foo::bar::Baz`, used in imports,
    /// visibilities or attributes.
    /// Technically, this variant is unnecessary and e.g., `Expr` can be used instead
    /// (paths in "mod" contexts have to be checked later for absence of generic arguments
    /// anyway, due to macros), but it is used to avoid weird suggestions about expected
    /// tokens when something goes wrong.
    Mod,
}

impl<'a> Parser<'a> {
    /// Parses a qualified path.
    /// Assumes that the leading `<` has been parsed already.
    ///
    /// `qualified_path = <type [as trait_ref]>::path`
    ///
    /// # Examples
    /// `<T>::default`
    /// `<T as U>::a`
    /// `<T as U>::F::a<S>` (without disambiguator)
    /// `<T as U>::F::a::<S>` (with disambiguator)
    pub(super) fn parse_qpath(&mut self, style: PathStyle) -> PResult<'a, (P<QSelf>, Path)> {
        let lo = self.prev_token.span;
        let ty = self.parse_ty()?;

        // `path` will contain the prefix of the path up to the `>`,
        // if any (e.g., `U` in the `<T as U>::*` examples
        // above). `path_span` has the span of that path, or an empty
        // span in the case of something like `<T>::Bar`.
        let (mut path, path_span);
        if self.eat_keyword(kw::As) {
            let path_lo = self.token.span;
            path = self.parse_path(PathStyle::Type)?;
            path_span = path_lo.to(self.prev_token.span);
        } else {
            path_span = self.token.span.to(self.token.span);
            path = ast::Path { segments: ThinVec::new(), span: path_span, tokens: None };
        }

        // See doc comment for `unmatched_angle_bracket_count`.
        self.expect(&token::Gt)?;
        if self.unmatched_angle_bracket_count > 0 {
            self.unmatched_angle_bracket_count -= 1;
            debug!("parse_qpath: (decrement) count={:?}", self.unmatched_angle_bracket_count);
        }

        if !self.recover_colon_before_qpath_proj() {
            self.expect(&token::ModSep)?;
        }

        let qself = P(QSelf { ty, path_span, position: path.segments.len() });
        self.parse_path_segments(&mut path.segments, style, None)?;

        Ok((
            qself,
            Path { segments: path.segments, span: lo.to(self.prev_token.span), tokens: None },
        ))
    }

    /// Recover from an invalid single colon, when the user likely meant a qualified path.
    /// We avoid emitting this if not followed by an identifier, as our assumption that the user
    /// intended this to be a qualified path may not be correct.
    ///
    /// ```ignore (diagnostics)
    /// <Bar as Baz<T>>:Qux
    ///                ^ help: use double colon
    /// ```
    fn recover_colon_before_qpath_proj(&mut self) -> bool {
        if !self.check_noexpect(&TokenKind::Colon)
            || self.look_ahead(1, |t| !t.is_ident() || t.is_reserved_ident())
        {
            return false;
        }

        self.bump(); // colon

        self.diagnostic()
            .struct_span_err(
                self.prev_token.span,
                "found single colon before projection in qualified path",
            )
            .span_suggestion(
                self.prev_token.span,
                "use double colon",
                "::",
                Applicability::MachineApplicable,
            )
            .emit();

        true
    }

    pub(super) fn parse_path(&mut self, style: PathStyle) -> PResult<'a, Path> {
        self.parse_path_inner(style, None)
    }

    /// Parses simple paths.
    ///
    /// `path = [::] segment+`
    /// `segment = ident | ident[::]<args> | ident[::](args) [-> type]`
    ///
    /// # Examples
    /// `a::b::C<D>` (without disambiguator)
    /// `a::b::C::<D>` (with disambiguator)
    /// `Fn(Args)` (without disambiguator)
    /// `Fn::(Args)` (with disambiguator)
    pub(super) fn parse_path_inner(
        &mut self,
        style: PathStyle,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Path> {
        let reject_generics_if_mod_style = |parser: &Parser<'_>, path: &Path| {
            // Ensure generic arguments don't end up in attribute paths, such as:
            //
            //     macro_rules! m {
            //         ($p:path) => { #[$p] struct S; }
            //     }
            //
            //     m!(inline<u8>); //~ ERROR: unexpected generic arguments in path
            //
            if style == PathStyle::Mod && path.segments.iter().any(|segment| segment.args.is_some())
            {
                parser
                    .struct_span_err(
                        path.segments
                            .iter()
                            .filter_map(|segment| segment.args.as_ref())
                            .map(|arg| arg.span())
                            .collect::<Vec<_>>(),
                        "unexpected generic arguments in path",
                    )
                    .emit();
            }
        };

        maybe_whole!(self, NtPath, |path| {
            reject_generics_if_mod_style(self, &path);
            path.into_inner()
        });

        if let token::Interpolated(nt) = &self.token.kind {
            if let token::NtTy(ty) = &**nt {
                if let ast::TyKind::Path(None, path) = &ty.kind {
                    let path = path.clone();
                    self.bump();
                    reject_generics_if_mod_style(self, &path);
                    return Ok(path);
                }
            }
        }

        let lo = self.token.span;
        let mut segments = ThinVec::new();
        let mod_sep_ctxt = self.token.span.ctxt();
        if self.eat(&token::ModSep) {
            segments.push(PathSegment::path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt)));
        }
        self.parse_path_segments(&mut segments, style, ty_generics)?;

        Ok(Path { segments, span: lo.to(self.prev_token.span), tokens: None })
    }

    pub(super) fn parse_path_segments(
        &mut self,
        segments: &mut ThinVec<PathSegment>,
        style: PathStyle,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, ()> {
        loop {
            let segment = self.parse_path_segment(style, ty_generics)?;
            if style == PathStyle::Expr {
                // In order to check for trailing angle brackets, we must have finished
                // recursing (`parse_path_segment` can indirectly call this function),
                // that is, the next token must be the highlighted part of the below example:
                //
                // `Foo::<Bar as Baz<T>>::Qux`
                //                      ^ here
                //
                // As opposed to the below highlight (if we had only finished the first
                // recursion):
                //
                // `Foo::<Bar as Baz<T>>::Qux`
                //                     ^ here
                //
                // `PathStyle::Expr` is only provided at the root invocation and never in
                // `parse_path_segment` to recurse and therefore can be checked to maintain
                // this invariant.
                self.check_trailing_angle_brackets(&segment, &[&token::ModSep]);
            }
            segments.push(segment);

            if self.is_import_coupler() || !self.eat(&token::ModSep) {
                return Ok(());
            }
        }
    }

    pub(super) fn parse_path_segment(
        &mut self,
        style: PathStyle,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, PathSegment> {
        let ident = self.parse_path_segment_ident()?;
        let is_args_start = |token: &Token| {
            matches!(
                token.kind,
                token::Lt
                    | token::BinOp(token::Shl)
                    | token::OpenDelim(Delimiter::Parenthesis)
                    | token::LArrow
            )
        };
        let check_args_start = |this: &mut Self| {
            this.expected_tokens.extend_from_slice(&[
                TokenType::Token(token::Lt),
                TokenType::Token(token::OpenDelim(Delimiter::Parenthesis)),
            ]);
            is_args_start(&this.token)
        };

        Ok(
            if style == PathStyle::Type && check_args_start(self)
                || style != PathStyle::Mod
                    && self.check(&token::ModSep)
                    && self.look_ahead(1, |t| is_args_start(t))
            {
                // We use `style == PathStyle::Expr` to check if this is in a recursion or not. If
                // it isn't, then we reset the unmatched angle bracket count as we're about to start
                // parsing a new path.
                if style == PathStyle::Expr {
                    self.unmatched_angle_bracket_count = 0;
                    self.max_angle_bracket_count = 0;
                }

                // Generic arguments are found - `<`, `(`, `::<` or `::(`.
                self.eat(&token::ModSep);
                let lo = self.token.span;
                let args = if self.eat_lt() {
                    // `<'a, T, A = U>`
                    let args = self.parse_angle_args_with_leading_angle_bracket_recovery(
                        style,
                        lo,
                        ty_generics,
                    )?;
                    self.expect_gt().map_err(|mut err| {
                        // Attempt to find places where a missing `>` might belong.
                        if let Some(arg) = args
                            .iter()
                            .rev()
                            .find(|arg| !matches!(arg, AngleBracketedArg::Constraint(_)))
                        {
                            err.span_suggestion_verbose(
                                arg.span().shrink_to_hi(),
                                "you might have meant to end the type parameters here",
                                ">",
                                Applicability::MaybeIncorrect,
                            );
                        }
                        err
                    })?;
                    let span = lo.to(self.prev_token.span);
                    AngleBracketedArgs { args, span }.into()
                } else {
                    // `(T, U) -> R`
                    let (inputs, _) = self.parse_paren_comma_seq(|p| p.parse_ty())?;
                    let inputs_span = lo.to(self.prev_token.span);
                    let output =
                        self.parse_ret_ty(AllowPlus::No, RecoverQPath::No, RecoverReturnSign::No)?;
                    let span = ident.span.to(self.prev_token.span);
                    ParenthesizedArgs { span, inputs, inputs_span, output }.into()
                };

                PathSegment { ident, args, id: ast::DUMMY_NODE_ID }
            } else {
                // Generic arguments are not found.
                PathSegment::from_ident(ident)
            },
        )
    }

    pub(super) fn parse_path_segment_ident(&mut self) -> PResult<'a, Ident> {
        match self.token.ident() {
            Some((ident, false)) if ident.is_path_segment_keyword() => {
                self.bump();
                Ok(ident)
            }
            _ => self.parse_ident(),
        }
    }

    /// Parses generic args (within a path segment) with recovery for extra leading angle brackets.
    /// For the purposes of understanding the parsing logic of generic arguments, this function
    /// can be thought of being the same as just calling `self.parse_angle_args()` if the source
    /// had the correct amount of leading angle brackets.
    ///
    /// ```ignore (diagnostics)
    /// bar::<<<<T as Foo>::Output>();
    ///      ^^ help: remove extra angle brackets
    /// ```
    fn parse_angle_args_with_leading_angle_bracket_recovery(
        &mut self,
        style: PathStyle,
        lo: Span,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Vec<AngleBracketedArg>> {
        // We need to detect whether there are extra leading left angle brackets and produce an
        // appropriate error and suggestion. This cannot be implemented by looking ahead at
        // upcoming tokens for a matching `>` character - if there are unmatched `<` tokens
        // then there won't be matching `>` tokens to find.
        //
        // To explain how this detection works, consider the following example:
        //
        // ```ignore (diagnostics)
        // bar::<<<<T as Foo>::Output>();
        //      ^^ help: remove extra angle brackets
        // ```
        //
        // Parsing of the left angle brackets starts in this function. We start by parsing the
        // `<` token (incrementing the counter of unmatched angle brackets on `Parser` via
        // `eat_lt`):
        //
        // *Upcoming tokens:* `<<<<T as Foo>::Output>;`
        // *Unmatched count:* 1
        // *`parse_path_segment` calls deep:* 0
        //
        // This has the effect of recursing as this function is called if a `<` character
        // is found within the expected generic arguments:
        //
        // *Upcoming tokens:* `<<<T as Foo>::Output>;`
        // *Unmatched count:* 2
        // *`parse_path_segment` calls deep:* 1
        //
        // Eventually we will have recursed until having consumed all of the `<` tokens and
        // this will be reflected in the count:
        //
        // *Upcoming tokens:* `T as Foo>::Output>;`
        // *Unmatched count:* 4
        // `parse_path_segment` calls deep:* 3
        //
        // The parser will continue until reaching the first `>` - this will decrement the
        // unmatched angle bracket count and return to the parent invocation of this function
        // having succeeded in parsing:
        //
        // *Upcoming tokens:* `::Output>;`
        // *Unmatched count:* 3
        // *`parse_path_segment` calls deep:* 2
        //
        // This will continue until the next `>` character which will also return successfully
        // to the parent invocation of this function and decrement the count:
        //
        // *Upcoming tokens:* `;`
        // *Unmatched count:* 2
        // *`parse_path_segment` calls deep:* 1
        //
        // At this point, this function will expect to find another matching `>` character but
        // won't be able to and will return an error. This will continue all the way up the
        // call stack until the first invocation:
        //
        // *Upcoming tokens:* `;`
        // *Unmatched count:* 2
        // *`parse_path_segment` calls deep:* 0
        //
        // In doing this, we have managed to work out how many unmatched leading left angle
        // brackets there are, but we cannot recover as the unmatched angle brackets have
        // already been consumed. To remedy this, we keep a snapshot of the parser state
        // before we do the above. We can then inspect whether we ended up with a parsing error
        // and unmatched left angle brackets and if so, restore the parser state before we
        // consumed any `<` characters to emit an error and consume the erroneous tokens to
        // recover by attempting to parse again.
        //
        // In practice, the recursion of this function is indirect and there will be other
        // locations that consume some `<` characters - as long as we update the count when
        // this happens, it isn't an issue.

        let is_first_invocation = style == PathStyle::Expr;
        // Take a snapshot before attempting to parse - we can restore this later.
        let snapshot = is_first_invocation.then(|| self.clone());

        debug!("parse_generic_args_with_leading_angle_bracket_recovery: (snapshotting)");
        match self.parse_angle_args(ty_generics) {
            Ok(args) => Ok(args),
            Err(e) if is_first_invocation && self.unmatched_angle_bracket_count > 0 => {
                // Swap `self` with our backup of the parser state before attempting to parse
                // generic arguments.
                let snapshot = mem::replace(self, snapshot.unwrap());

                // Eat the unmatched angle brackets.
                let all_angle_brackets = (0..snapshot.unmatched_angle_bracket_count)
                    .fold(true, |a, _| a && self.eat_lt());

                if !all_angle_brackets {
                    // If there are other tokens in between the extraneous `<`s, we cannot simply
                    // suggest to remove them. This check also prevents us from accidentally ending
                    // up in the middle of a multibyte character (issue #84104).
                    let _ = mem::replace(self, snapshot);
                    Err(e)
                } else {
                    // Cancel error from being unable to find `>`. We know the error
                    // must have been this due to a non-zero unmatched angle bracket
                    // count.
                    e.cancel();

                    debug!(
                        "parse_generic_args_with_leading_angle_bracket_recovery: (snapshot failure) \
                         snapshot.count={:?}",
                        snapshot.unmatched_angle_bracket_count,
                    );

                    // Make a span over ${unmatched angle bracket count} characters.
                    // This is safe because `all_angle_brackets` ensures that there are only `<`s,
                    // i.e. no multibyte characters, in this range.
                    let span =
                        lo.with_hi(lo.lo() + BytePos(snapshot.unmatched_angle_bracket_count));
                    self.struct_span_err(
                        span,
                        &format!(
                            "unmatched angle bracket{}",
                            pluralize!(snapshot.unmatched_angle_bracket_count)
                        ),
                    )
                    .span_suggestion(
                        span,
                        &format!(
                            "remove extra angle bracket{}",
                            pluralize!(snapshot.unmatched_angle_bracket_count)
                        ),
                        "",
                        Applicability::MachineApplicable,
                    )
                    .emit();

                    // Try again without unmatched angle bracket characters.
                    self.parse_angle_args(ty_generics)
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Parses (possibly empty) list of generic arguments / associated item constraints,
    /// possibly including trailing comma.
    pub(super) fn parse_angle_args(
        &mut self,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Vec<AngleBracketedArg>> {
        let mut args = Vec::new();
        while let Some(arg) = self.parse_angle_arg(ty_generics)? {
            args.push(arg);
            if !self.eat(&token::Comma) {
                if self.check_noexpect(&TokenKind::Semi)
                    && self.look_ahead(1, |t| t.is_ident() || t.is_lifetime())
                {
                    // Add `>` to the list of expected tokens.
                    self.check(&token::Gt);
                    // Handle `,` to `;` substitution
                    let mut err = self.unexpected::<()>().unwrap_err();
                    self.bump();
                    err.span_suggestion_verbose(
                        self.prev_token.span.until(self.token.span),
                        "use a comma to separate type parameters",
                        ", ",
                        Applicability::MachineApplicable,
                    );
                    err.emit();
                    continue;
                }
                if !self.token.kind.should_end_const_arg() {
                    if self.handle_ambiguous_unbraced_const_arg(&mut args)? {
                        // We've managed to (partially) recover, so continue trying to parse
                        // arguments.
                        continue;
                    }
                }
                break;
            }
        }
        Ok(args)
    }

    /// Parses a single argument in the angle arguments `<...>` of a path segment.
    fn parse_angle_arg(
        &mut self,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Option<AngleBracketedArg>> {
        let lo = self.token.span;
        let arg = self.parse_generic_arg(ty_generics)?;
        match arg {
            Some(arg) => {
                // we are using noexpect here because we first want to find out if either `=` or `:`
                // is present and then use that info to push the other token onto the tokens list
                let separated =
                    self.check_noexpect(&token::Colon) || self.check_noexpect(&token::Eq);
                if separated && (self.check(&token::Colon) | self.check(&token::Eq)) {
                    let arg_span = arg.span();
                    let (binder, ident, gen_args) = match self.get_ident_from_generic_arg(&arg) {
                        Ok(ident_gen_args) => ident_gen_args,
                        Err(()) => return Ok(Some(AngleBracketedArg::Arg(arg))),
                    };
                    if binder {
                        // FIXME(compiler-errors): this could be improved by suggesting lifting
                        // this up to the trait, at least before this becomes real syntax.
                        // e.g. `Trait<for<'a> Assoc = Ty>` -> `for<'a> Trait<Assoc = Ty>`
                        return Err(self.struct_span_err(
                            arg_span,
                            "`for<...>` is not allowed on associated type bounds",
                        ));
                    }
                    let kind = if self.eat(&token::Colon) {
                        // Parse associated type constraint bound.

                        let bounds = self.parse_generic_bounds(Some(self.prev_token.span))?;
                        AssocConstraintKind::Bound { bounds }
                    } else if self.eat(&token::Eq) {
                        self.parse_assoc_equality_term(ident, self.prev_token.span)?
                    } else {
                        unreachable!();
                    };

                    let span = lo.to(self.prev_token.span);

                    // Gate associated type bounds, e.g., `Iterator<Item: Ord>`.
                    if let AssocConstraintKind::Bound { .. } = kind {
                        self.sess.gated_spans.gate(sym::associated_type_bounds, span);
                    }
                    let constraint =
                        AssocConstraint { id: ast::DUMMY_NODE_ID, ident, gen_args, kind, span };
                    Ok(Some(AngleBracketedArg::Constraint(constraint)))
                } else {
                    // we only want to suggest `:` and `=` in contexts where the previous token
                    // is an ident and the current token or the next token is an ident
                    if self.prev_token.is_ident()
                        && (self.token.is_ident() || self.look_ahead(1, |token| token.is_ident()))
                    {
                        self.check(&token::Colon);
                        self.check(&token::Eq);
                    }
                    Ok(Some(AngleBracketedArg::Arg(arg)))
                }
            }
            _ => Ok(None),
        }
    }

    /// Parse the term to the right of an associated item equality constraint.
    /// That is, parse `<term>` in `Item = <term>`.
    /// Right now, this only admits types in `<term>`.
    fn parse_assoc_equality_term(
        &mut self,
        ident: Ident,
        eq: Span,
    ) -> PResult<'a, AssocConstraintKind> {
        let arg = self.parse_generic_arg(None)?;
        let span = ident.span.to(self.prev_token.span);
        let term = match arg {
            Some(GenericArg::Type(ty)) => ty.into(),
            Some(GenericArg::Const(c)) => {
                self.sess.gated_spans.gate(sym::associated_const_equality, span);
                c.into()
            }
            Some(GenericArg::Lifetime(lt)) => {
                self.struct_span_err(span, "associated lifetimes are not supported")
                    .span_label(lt.ident.span, "the lifetime is given here")
                    .help("if you meant to specify a trait object, write `dyn Trait + 'lifetime`")
                    .emit();
                self.mk_ty(span, ast::TyKind::Err).into()
            }
            None => {
                let after_eq = eq.shrink_to_hi();
                let before_next = self.token.span.shrink_to_lo();
                let mut err = self
                    .struct_span_err(after_eq.to(before_next), "missing type to the right of `=`");
                if matches!(self.token.kind, token::Comma | token::Gt) {
                    err.span_suggestion(
                        self.sess.source_map().next_point(eq).to(before_next),
                        "to constrain the associated type, add a type after `=`",
                        " TheType",
                        Applicability::HasPlaceholders,
                    );
                    err.span_suggestion(
                        eq.to(before_next),
                        &format!("remove the `=` if `{}` is a type", ident),
                        "",
                        Applicability::MaybeIncorrect,
                    )
                } else {
                    err.span_label(
                        self.token.span,
                        &format!("expected type, found {}", super::token_descr(&self.token)),
                    )
                };
                return Err(err);
            }
        };
        Ok(AssocConstraintKind::Equality { term })
    }

    /// We do not permit arbitrary expressions as const arguments. They must be one of:
    /// - An expression surrounded in `{}`.
    /// - A literal.
    /// - A numeric literal prefixed by `-`.
    /// - A single-segment path.
    pub(super) fn expr_is_valid_const_arg(&self, expr: &P<rustc_ast::Expr>) -> bool {
        match &expr.kind {
            ast::ExprKind::Block(_, _)
            | ast::ExprKind::Lit(_)
            | ast::ExprKind::IncludedBytes(..) => true,
            ast::ExprKind::Unary(ast::UnOp::Neg, expr) => {
                matches!(expr.kind, ast::ExprKind::Lit(_))
            }
            // We can only resolve single-segment paths at the moment, because multi-segment paths
            // require type-checking: see `visit_generic_arg` in `src/librustc_resolve/late.rs`.
            ast::ExprKind::Path(None, path)
                if path.segments.len() == 1 && path.segments[0].args.is_none() =>
            {
                true
            }
            _ => false,
        }
    }

    /// Parse a const argument, e.g. `<3>`. It is assumed the angle brackets will be parsed by
    /// the caller.
    pub(super) fn parse_const_arg(&mut self) -> PResult<'a, AnonConst> {
        // Parse const argument.
        let value = if let token::OpenDelim(Delimiter::Brace) = self.token.kind {
            self.parse_block_expr(None, self.token.span, BlockCheckMode::Default)?
        } else {
            self.handle_unambiguous_unbraced_const_arg()?
        };
        Ok(AnonConst { id: ast::DUMMY_NODE_ID, value })
    }

    /// Parse a generic argument in a path segment.
    /// This does not include constraints, e.g., `Item = u8`, which is handled in `parse_angle_arg`.
    pub(super) fn parse_generic_arg(
        &mut self,
        ty_generics: Option<&Generics>,
    ) -> PResult<'a, Option<GenericArg>> {
        let start = self.token.span;
        let arg = if self.check_lifetime() && self.look_ahead(1, |t| !t.is_like_plus()) {
            // Parse lifetime argument.
            GenericArg::Lifetime(self.expect_lifetime())
        } else if self.check_const_arg() {
            // Parse const argument.
            GenericArg::Const(self.parse_const_arg()?)
        } else if self.check_type() {
            // Parse type argument.

            // Proactively create a parser snapshot enabling us to rewind and try to reparse the
            // input as a const expression in case we fail to parse a type. If we successfully
            // do so, we will report an error that it needs to be wrapped in braces.
            let mut snapshot = None;
            if self.may_recover() && self.token.can_begin_expr() {
                snapshot = Some(self.create_snapshot_for_diagnostic());
            }

            match self.parse_ty() {
                Ok(ty) => {
                    // Since the type parser recovers from some malformed slice and array types and
                    // successfully returns a type, we need to look for `TyKind::Err`s in the
                    // type to determine if error recovery has occurred and if the input is not a
                    // syntactically valid type after all.
                    if let ast::TyKind::Slice(inner_ty) | ast::TyKind::Array(inner_ty, _) = &ty.kind
                        && let ast::TyKind::Err = inner_ty.kind
                        && let Some(snapshot) = snapshot
                        && let Some(expr) = self.recover_unbraced_const_arg_that_can_begin_ty(snapshot)
                    {
                        return Ok(Some(self.dummy_const_arg_needs_braces(
                            self.struct_span_err(expr.span, "invalid const generic expression"),
                            expr.span,
                        )));
                    }

                    GenericArg::Type(ty)
                }
                Err(err) => {
                    if let Some(snapshot) = snapshot
                        && let Some(expr) = self.recover_unbraced_const_arg_that_can_begin_ty(snapshot)
                    {
                        return Ok(Some(self.dummy_const_arg_needs_braces(
                            err,
                            expr.span,
                        )));
                    }
                    // Try to recover from possible `const` arg without braces.
                    return self.recover_const_arg(start, err).map(Some);
                }
            }
        } else if self.token.is_keyword(kw::Const) {
            return self.recover_const_param_declaration(ty_generics);
        } else {
            // Fall back by trying to parse a const-expr expression. If we successfully do so,
            // then we should report an error that it needs to be wrapped in braces.
            let snapshot = self.create_snapshot_for_diagnostic();
            match self.parse_expr_res(Restrictions::CONST_EXPR, None) {
                Ok(expr) => {
                    return Ok(Some(self.dummy_const_arg_needs_braces(
                        self.struct_span_err(expr.span, "invalid const generic expression"),
                        expr.span,
                    )));
                }
                Err(err) => {
                    self.restore_snapshot(snapshot);
                    err.cancel();
                    return Ok(None);
                }
            }
        };
        Ok(Some(arg))
    }

    /// Given a arg inside of generics, we try to destructure it as if it were the LHS in
    /// `LHS = ...`, i.e. an associated type binding.
    /// This returns a bool indicating if there are any `for<'a, 'b>` binder args, the
    /// identifier, and any GAT arguments.
    fn get_ident_from_generic_arg(
        &self,
        gen_arg: &GenericArg,
    ) -> Result<(bool, Ident, Option<GenericArgs>), ()> {
        if let GenericArg::Type(ty) = gen_arg {
            if let ast::TyKind::Path(qself, path) = &ty.kind
                && qself.is_none()
                && let [seg] = path.segments.as_slice()
            {
                return Ok((false, seg.ident, seg.args.as_deref().cloned()));
            } else if let ast::TyKind::TraitObject(bounds, ast::TraitObjectSyntax::None) = &ty.kind
                && let [ast::GenericBound::Trait(trait_ref, ast::TraitBoundModifier::None)] =
                    bounds.as_slice()
                && let [seg] = trait_ref.trait_ref.path.segments.as_slice()
            {
                return Ok((true, seg.ident, seg.args.as_deref().cloned()));
            }
        }
        Err(())
    }
}
