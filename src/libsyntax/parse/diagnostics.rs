use crate::ast;
use crate::ast::{Expr, ExprKind, Item, ItemKind, Pat, PatKind, QSelf, Ty, TyKind};
use crate::parse::parser::PathStyle;
use crate::parse::token;
use crate::parse::PResult;
use crate::parse::Parser;
use crate::print::pprust;
use crate::ptr::P;
use crate::ThinVec;
use errors::Applicability;
use syntax_pos::Span;

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
            span: syntax_pos::DUMMY_SP,
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
}
