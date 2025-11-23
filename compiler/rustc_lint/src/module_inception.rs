//! Lint for detecting modules that have the same name as their parent module.

use rustc_hir::{Body, Item, ItemKind};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::Symbol;

use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `module_inception` lint detects modules that have the same name as their parent module.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// mod foo {
    ///     mod foo {
    ///         pub fn bar() {}
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// A typical beginner mistake is to have `mod foo;` and again `mod foo { .. }`
    /// in `foo.rs`. The expectation is that items inside the inner `mod foo { .. }`
    /// are then available through `foo::x`, but they are only available through
    /// `foo::foo::x`. If this is done on purpose, it would be better to choose a
    /// more representative module name.
    pub MODULE_INCEPTION,
    Warn,
    "modules that have the same name as their parent module"
}

struct ModInfo {
    name: Symbol,
    /// How many bodies are between this module and the current lint pass position.
    ///
    /// Only the most recently seen module is updated when entering/exiting a body.
    in_body_count: u32,
}

pub(crate) struct ModuleInception {
    /// The module path the lint pass is in.
    modules: Vec<ModInfo>,
}

impl ModuleInception {
    pub(crate) fn new() -> Self {
        Self { modules: Vec::new() }
    }
}

impl_lint_pass!(ModuleInception => [MODULE_INCEPTION]);

impl<'tcx> LateLintPass<'tcx> for ModuleInception {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if let ItemKind::Mod(ident, _) = item.kind {
            // Check if this module has the same name as its parent module
            if let [.., prev] = &*self.modules
                && prev.name == ident.name
                && prev.in_body_count == 0
                && !item.span.from_expansion()
            {
                cx.span_lint(MODULE_INCEPTION, item.span, |lint| {
                    lint.primary_message("module has the same name as its containing module");
                });
            }

            self.modules.push(ModInfo { name: ident.name, in_body_count: 0 });
        }
    }

    fn check_item_post(&mut self, _cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if matches!(item.kind, ItemKind::Mod(..)) {
            self.modules.pop();
        }
    }

    fn check_body(&mut self, _: &LateContext<'tcx>, _: &Body<'tcx>) {
        if let [.., last] = &mut *self.modules {
            last.in_body_count += 1;
        }
    }

    fn check_body_post(&mut self, _: &LateContext<'tcx>, _: &Body<'tcx>) {
        if let [.., last] = &mut *self.modules {
            last.in_body_count -= 1;
        }
    }
}
