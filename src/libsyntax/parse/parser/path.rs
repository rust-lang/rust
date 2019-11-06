use super::{Parser, PResult, TokenType};

use crate::{maybe_whole, ThinVec};
use crate::ast::{self, QSelf, Path, PathSegment, Ident, ParenthesizedArgs, AngleBracketedArgs};
use crate::ast::{AnonConst, GenericArg, AssocTyConstraint, AssocTyConstraintKind, BlockCheckMode};
use crate::parse::token::{self, Token};
use crate::source_map::{Span, BytePos};
use crate::symbol::kw;

use std::mem;
use log::debug;
use errors::{Applicability, pluralize};

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
    pub(super) fn parse_qpath(&mut self, style: PathStyle) -> PResult<'a, (QSelf, Path)> {
        let lo = self.prev_span;
        let ty = self.parse_ty()?;

        // `path` will contain the prefix of the path up to the `>`,
        // if any (e.g., `U` in the `<T as U>::*` examples
        // above). `path_span` has the span of that path, or an empty
        // span in the case of something like `<T>::Bar`.
        let (mut path, path_span);
        if self.eat_keyword(kw::As) {
            let path_lo = self.token.span;
            path = self.parse_path(PathStyle::Type)?;
            path_span = path_lo.to(self.prev_span);
        } else {
            path_span = self.token.span.to(self.token.span);
            path = ast::Path { segments: Vec::new(), span: path_span };
        }

        // See doc comment for `unmatched_angle_bracket_count`.
        self.expect(&token::Gt)?;
        if self.unmatched_angle_bracket_count > 0 {
            self.unmatched_angle_bracket_count -= 1;
            debug!("parse_qpath: (decrement) count={:?}", self.unmatched_angle_bracket_count);
        }

        self.expect(&token::ModSep)?;

        let qself = QSelf { ty, path_span, position: path.segments.len() };
        self.parse_path_segments(&mut path.segments, style)?;

        Ok((qself, Path { segments: path.segments, span: lo.to(self.prev_span) }))
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
    pub fn parse_path(&mut self, style: PathStyle) -> PResult<'a, Path> {
        maybe_whole!(self, NtPath, |path| {
            if style == PathStyle::Mod &&
               path.segments.iter().any(|segment| segment.args.is_some()) {
                self.diagnostic().span_err(path.span, "unexpected generic arguments in path");
            }
            path
        });

        let lo = self.meta_var_span.unwrap_or(self.token.span);
        let mut segments = Vec::new();
        let mod_sep_ctxt = self.token.span.ctxt();
        if self.eat(&token::ModSep) {
            segments.push(PathSegment::path_root(lo.shrink_to_lo().with_ctxt(mod_sep_ctxt)));
        }
        self.parse_path_segments(&mut segments, style)?;

        Ok(Path { segments, span: lo.to(self.prev_span) })
    }

    /// Like `parse_path`, but also supports parsing `Word` meta items into paths for
    /// backwards-compatibility. This is used when parsing derive macro paths in `#[derive]`
    /// attributes.
    fn parse_path_allowing_meta(&mut self, style: PathStyle) -> PResult<'a, Path> {
        let meta_ident = match self.token.kind {
            token::Interpolated(ref nt) => match **nt {
                token::NtMeta(ref item) => match item.tokens.is_empty() {
                    true => Some(item.path.clone()),
                    false => None,
                },
                _ => None,
            },
            _ => None,
        };
        if let Some(path) = meta_ident {
            self.bump();
            return Ok(path);
        }
        self.parse_path(style)
    }

    /// Parse a list of paths inside `#[derive(path_0, ..., path_n)]`.
    pub fn parse_derive_paths(&mut self) -> PResult<'a, Vec<Path>> {
        self.expect(&token::OpenDelim(token::Paren))?;
        let mut list = Vec::new();
        while !self.eat(&token::CloseDelim(token::Paren)) {
            let path = self.parse_path_allowing_meta(PathStyle::Mod)?;
            list.push(path);
            if !self.eat(&token::Comma) {
                self.expect(&token::CloseDelim(token::Paren))?;
                break
            }
        }
        Ok(list)
    }

    pub(super) fn parse_path_segments(
        &mut self,
        segments: &mut Vec<PathSegment>,
        style: PathStyle,
    ) -> PResult<'a, ()> {
        loop {
            let segment = self.parse_path_segment(style)?;
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
                self.check_trailing_angle_brackets(&segment, token::ModSep);
            }
            segments.push(segment);

            if self.is_import_coupler() || !self.eat(&token::ModSep) {
                return Ok(());
            }
        }
    }

    pub(super) fn parse_path_segment(&mut self, style: PathStyle) -> PResult<'a, PathSegment> {
        let ident = self.parse_path_segment_ident()?;

        let is_args_start = |token: &Token| match token.kind {
            token::Lt | token::BinOp(token::Shl) | token::OpenDelim(token::Paren)
            | token::LArrow => true,
            _ => false,
        };
        let check_args_start = |this: &mut Self| {
            this.expected_tokens.extend_from_slice(
                &[TokenType::Token(token::Lt), TokenType::Token(token::OpenDelim(token::Paren))]
            );
            is_args_start(&this.token)
        };

        Ok(if style == PathStyle::Type && check_args_start(self) ||
              style != PathStyle::Mod && self.check(&token::ModSep)
                                      && self.look_ahead(1, |t| is_args_start(t)) {
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
                let (args, constraints) =
                    self.parse_generic_args_with_leaning_angle_bracket_recovery(style, lo)?;
                self.expect_gt()?;
                let span = lo.to(self.prev_span);
                AngleBracketedArgs { args, constraints, span }.into()
            } else {
                // `(T, U) -> R`
                let (inputs, _) = self.parse_paren_comma_seq(|p| p.parse_ty())?;
                let span = ident.span.to(self.prev_span);
                let output = if self.eat(&token::RArrow) {
                    Some(self.parse_ty_common(false, false, false)?)
                } else {
                    None
                };
                ParenthesizedArgs { inputs, output, span }.into()
            };

            PathSegment { ident, args, id: ast::DUMMY_NODE_ID }
        } else {
            // Generic arguments are not found.
            PathSegment::from_ident(ident)
        })
    }

    pub(super) fn parse_path_segment_ident(&mut self) -> PResult<'a, Ident> {
        match self.token.kind {
            token::Ident(name, _) if name.is_path_segment_keyword() => {
                let span = self.token.span;
                self.bump();
                Ok(Ident::new(name, span))
            }
            _ => self.parse_ident(),
        }
    }

    /// Parses generic args (within a path segment) with recovery for extra leading angle brackets.
    /// For the purposes of understanding the parsing logic of generic arguments, this function
    /// can be thought of being the same as just calling `self.parse_generic_args()` if the source
    /// had the correct amount of leading angle brackets.
    ///
    /// ```ignore (diagnostics)
    /// bar::<<<<T as Foo>::Output>();
    ///      ^^ help: remove extra angle brackets
    /// ```
    fn parse_generic_args_with_leaning_angle_bracket_recovery(
        &mut self,
        style: PathStyle,
        lo: Span,
    ) -> PResult<'a, (Vec<GenericArg>, Vec<AssocTyConstraint>)> {
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
        let snapshot = if is_first_invocation {
            Some(self.clone())
        } else {
            None
        };

        debug!("parse_generic_args_with_leading_angle_bracket_recovery: (snapshotting)");
        match self.parse_generic_args() {
            Ok(value) => Ok(value),
            Err(ref mut e) if is_first_invocation && self.unmatched_angle_bracket_count > 0 => {
                // Cancel error from being unable to find `>`. We know the error
                // must have been this due to a non-zero unmatched angle bracket
                // count.
                e.cancel();

                // Swap `self` with our backup of the parser state before attempting to parse
                // generic arguments.
                let snapshot = mem::replace(self, snapshot.unwrap());

                debug!(
                    "parse_generic_args_with_leading_angle_bracket_recovery: (snapshot failure) \
                     snapshot.count={:?}",
                    snapshot.unmatched_angle_bracket_count,
                );

                // Eat the unmatched angle brackets.
                for _ in 0..snapshot.unmatched_angle_bracket_count {
                    self.eat_lt();
                }

                // Make a span over ${unmatched angle bracket count} characters.
                let span = lo.with_hi(
                    lo.lo() + BytePos(snapshot.unmatched_angle_bracket_count)
                );
                self.diagnostic()
                    .struct_span_err(
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
                        String::new(),
                        Applicability::MachineApplicable,
                    )
                    .emit();

                // Try again without unmatched angle bracket characters.
                self.parse_generic_args()
            },
            Err(e) => Err(e),
        }
    }

    /// Parses (possibly empty) list of lifetime and type arguments and associated type bindings,
    /// possibly including trailing comma.
    fn parse_generic_args(&mut self) -> PResult<'a, (Vec<GenericArg>, Vec<AssocTyConstraint>)> {
        let mut args = Vec::new();
        let mut constraints = Vec::new();
        let mut misplaced_assoc_ty_constraints: Vec<Span> = Vec::new();
        let mut assoc_ty_constraints: Vec<Span> = Vec::new();

        let args_lo = self.token.span;

        loop {
            if self.check_lifetime() && self.look_ahead(1, |t| !t.is_like_plus()) {
                // Parse lifetime argument.
                args.push(GenericArg::Lifetime(self.expect_lifetime()));
                misplaced_assoc_ty_constraints.append(&mut assoc_ty_constraints);
            } else if self.check_ident()
                && self.look_ahead(1, |t| t == &token::Eq || t == &token::Colon)
            {
                // Parse associated type constraint.
                let lo = self.token.span;
                let ident = self.parse_ident()?;
                let kind = if self.eat(&token::Eq) {
                    AssocTyConstraintKind::Equality {
                        ty: self.parse_ty()?,
                    }
                } else if self.eat(&token::Colon) {
                    AssocTyConstraintKind::Bound {
                        bounds: self.parse_generic_bounds(Some(self.prev_span))?,
                    }
                } else {
                    unreachable!();
                };

                let span = lo.to(self.prev_span);

                // Gate associated type bounds, e.g., `Iterator<Item: Ord>`.
                if let AssocTyConstraintKind::Bound { .. } = kind {
                    self.sess.gated_spans.associated_type_bounds.borrow_mut().push(span);
                }

                constraints.push(AssocTyConstraint {
                    id: ast::DUMMY_NODE_ID,
                    ident,
                    kind,
                    span,
                });
                assoc_ty_constraints.push(span);
            } else if self.check_const_arg() {
                // Parse const argument.
                let expr = if let token::OpenDelim(token::Brace) = self.token.kind {
                    self.parse_block_expr(
                        None, self.token.span, BlockCheckMode::Default, ThinVec::new()
                    )?
                } else if self.token.is_ident() {
                    // FIXME(const_generics): to distinguish between idents for types and consts,
                    // we should introduce a GenericArg::Ident in the AST and distinguish when
                    // lowering to the HIR. For now, idents for const args are not permitted.
                    if self.token.is_bool_lit() {
                        self.parse_literal_maybe_minus()?
                    } else {
                        return Err(
                            self.fatal("identifiers may currently not be used for const generics")
                        );
                    }
                } else {
                    self.parse_literal_maybe_minus()?
                };
                let value = AnonConst {
                    id: ast::DUMMY_NODE_ID,
                    value: expr,
                };
                args.push(GenericArg::Const(value));
                misplaced_assoc_ty_constraints.append(&mut assoc_ty_constraints);
            } else if self.check_type() {
                // Parse type argument.
                args.push(GenericArg::Type(self.parse_ty()?));
                misplaced_assoc_ty_constraints.append(&mut assoc_ty_constraints);
            } else {
                break
            }

            if !self.eat(&token::Comma) {
                break
            }
        }

        // FIXME: we would like to report this in ast_validation instead, but we currently do not
        // preserve ordering of generic parameters with respect to associated type binding, so we
        // lose that information after parsing.
        if misplaced_assoc_ty_constraints.len() > 0 {
            let mut err = self.struct_span_err(
                args_lo.to(self.prev_span),
                "associated type bindings must be declared after generic parameters",
            );
            for span in misplaced_assoc_ty_constraints {
                err.span_label(
                    span,
                    "this associated type binding should be moved after the generic parameters",
                );
            }
            err.emit();
        }

        Ok((args, constraints))
    }
}
