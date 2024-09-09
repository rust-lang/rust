use rustc_hir::{
    ImplItem, ImplItemKind, Item, ItemKind, Lifetime, LifetimeName, TraitItem, TraitItemKind,
};
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::symbol::kw;

use crate::lints::{ElidedNamedLifetime, ElidedNamedLifetimeSuggestion};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `elided_named_lifetimes` lint detects when an elided
    /// lifetime ends up being a named lifetime, such as `'static`
    /// or some lifetime parameter `'a`.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(elided_named_lifetimes)]
    /// struct Foo;
    /// impl Foo {
    ///     pub fn get_mut(&'static self, x: &mut u8) -> &mut u8 {
    ///         unsafe { &mut *(x as *mut _) }
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Lifetime elision is quite useful, because it frees you from having
    /// to give each lifetime its own name, but sometimes it can produce
    /// somewhat surprising resolutions. In safe code, it is mostly okay,
    /// because the borrow checker prevents any unsoundness, so the worst
    /// case scenario is you get a confusing error message in some other place.
    /// But with `unsafe` code, such unexpected resolutions may lead to unsound code.
    pub ELIDED_NAMED_LIFETIMES,
    Warn,
    "detects when an elided lifetime gets resolved to be `'static` or some named parameter"
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ElidedNamedLifetimes {
    allow_static: bool,
}

impl_lint_pass!(ElidedNamedLifetimes => [ELIDED_NAMED_LIFETIMES]);

impl<'tcx> LateLintPass<'tcx> for ElidedNamedLifetimes {
    fn check_trait_item(&mut self, _: &LateContext<'tcx>, item: &'tcx TraitItem<'tcx>) {
        if let TraitItemKind::Const(..) = item.kind {
            self.allow_static = true;
        }
    }
    fn check_trait_item_post(&mut self, _: &LateContext<'tcx>, _: &'tcx TraitItem<'tcx>) {
        self.allow_static = false;
    }

    fn check_impl_item(&mut self, _: &LateContext<'tcx>, item: &'tcx ImplItem<'tcx>) {
        if let ImplItemKind::Const(..) = item.kind {
            self.allow_static = true;
        }
    }
    fn check_impl_item_post(&mut self, _: &LateContext<'tcx>, _: &'tcx ImplItem<'tcx>) {
        self.allow_static = false;
    }

    fn check_item(&mut self, _: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Const(..) | ItemKind::Static(..) = item.kind {
            self.allow_static = true;
        }
    }
    fn check_item_post(&mut self, _: &LateContext<'tcx>, _: &'tcx Item<'tcx>) {
        self.allow_static = false;
    }

    fn check_lifetime(&mut self, cx: &LateContext<'tcx>, lifetime: &'tcx Lifetime) {
        // `.is_elided()` should probably be called `.resolves_to_elided()`,
        // and `.is_anonymous()` is actually the thing that we need here.
        if !lifetime.is_anonymous() {
            return;
        }
        let (name, declaration) = match lifetime.res {
            LifetimeName::Param(param) => {
                let name = cx.tcx().item_name(param.into());
                if name == kw::UnderscoreLifetime {
                    return;
                }
                let span = cx.tcx().def_span(param);
                (name, Some(span))
            }
            LifetimeName::Static => {
                if self.allow_static {
                    return;
                }
                (kw::StaticLifetime, None)
            }
            LifetimeName::ImplicitObjectLifetimeDefault
            | LifetimeName::Error
            | LifetimeName::Infer => return,
        };
        cx.emit_lint(
            ELIDED_NAMED_LIFETIMES,
            ElidedNamedLifetime {
                span: lifetime.ident.span,
                sugg: {
                    let (span, code) = lifetime.suggestion(name.as_str());
                    ElidedNamedLifetimeSuggestion { span, code }
                },
                name,
                declaration,
            },
        )
    }
}
