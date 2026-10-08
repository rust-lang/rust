use rustc_ast::{self as ast, Path, TyKind};
use rustc_errors::{Applicability, PResult};
use rustc_span::{Ident, Span, kw, sym};
use thin_vec::{ThinVec, thin_vec};

use super::AllowPlus;
use crate::diagnostics;
use crate::parser::{Parser, PathStyle, token_descr};

impl<'a> Parser<'a> {
    pub(super) fn report_unexpected_token_in_ty(&mut self, lo: Span) -> PResult<'a, TyKind> {
        let msg = format!("expected type, found {}", token_descr(&self.token));
        let mut err = self.dcx().struct_span_err(lo, msg);
        err.span_label(lo, "expected type");

        if self.may_recover()
            && (self.eat_keyword_noexpect(kw::True) || self.eat_keyword_noexpect(kw::False))
        {
            err.span_suggestion(
                self.prev_token.span,
                "the type is called",
                "bool",
                Applicability::MachineApplicable,
            );
            err.emit();
            let path = Path::from_ident(Ident { span: self.prev_token.span, name: sym::bool });
            return Ok(TyKind::Path(None, path));
        }

        Err(err)
    }

    /// Recover from `for<…> dyn Trait` or `for<…> impl Trait`.
    pub(super) fn recover_from_binder_before_dyn_or_impl_ty(
        &mut self,
        bound_vars: &mut ThinVec<ast::GenericParam>,
        allow_plus: AllowPlus,
        lo: Span,
    ) -> PResult<'a, Option<TyKind>> {
        if !self.may_recover() || !self.token.is_keyword(kw::Impl) && !self.can_begin_dyn_ty() {
            return Ok(None);
        }
        self.bump();
        let (kw, _) = self.prev_token.ident().unwrap();
        let removal_span = kw.span.with_hi(self.token.span.lo());
        let path = self.parse_path(PathStyle::Type)?;
        let mut bounds = thin_vec![ast::GenericBound::Trait(ast::PolyTraitRef::new(
            std::mem::take(bound_vars),
            path,
            ast::TraitBoundModifiers::NONE,
            lo.to(self.prev_token.span),
            ast::Parens::No,
        ))];
        if allow_plus == AllowPlus::Yes && self.check_plus() {
            self.eat_plus();
            bounds.append(&mut self.parse_generic_bounds()?);
        }
        self.dcx().emit_err(diagnostics::TransposeDynOrImpl {
            span: kw.span,
            kw: kw.name.as_str(),
            sugg: diagnostics::TransposeDynOrImplSugg {
                removal_span,
                insertion_span: lo.shrink_to_lo(),
                kw: kw.name.as_str(),
            },
        });
        Ok(Some(match kw.name {
            kw::Dyn => TyKind::TraitObject(bounds, ast::TraitObjectSyntax::Dyn),
            kw::Impl => TyKind::ImplTrait(ast::DUMMY_NODE_ID, bounds),
            _ => unreachable!(),
        }))
    }
}
