use rustc_data_structures::fx::FxIndexMap;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{self as hir, LifetimeSource, LifetimeSyntax};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;
use tracing::instrument;

use crate::{LateContext, LateLintPass, LintContext, lints};

declare_lint! {
    /// The `mismatched_lifetime_syntaxes` lint detects when an
    /// elided lifetime uses different syntax between function
    /// arguments and return values.
    ///
    /// The three kinds of syntax are:
    /// 1. Named lifetimes, such as `'static` or `'a`.
    /// 2. The anonymous lifetime `'_`.
    /// 3. Hidden lifetimes, such as `&u8` or `ThisHasALifetimeGeneric`.
    ///
    /// As an exception, this lint allows references with hidden or
    /// anonymous lifetimes to be paired with paths using anonymous
    /// lifetimes.
    ///
    /// ### Example
    ///
    /// ```rust,compile_fail
    /// #![deny(mismatched_lifetime_syntaxes)]
    ///
    /// pub fn mixing_named_with_hidden(v: &'static u8) -> &u8 {
    ///     v
    /// }
    ///
    /// struct Person<'a> {
    ///     name: &'a str,
    /// }
    ///
    /// pub fn mixing_hidden_with_anonymous(v: Person) -> Person<'_> {
    ///     v
    /// }
    ///
    /// struct Foo;
    ///
    /// impl Foo {
    ///     // Lifetime elision results in the output lifetime becoming
    ///     // `'static`, which is not what was intended.
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
    /// Lifetime elision is useful because it frees you from having to
    /// give each lifetime its own name and explicitly show the
    /// relation of input and output lifetimes for common
    /// cases. However, a lifetime that uses inconsistent syntax
    /// between related arguments and return values is more confusing.
    ///
    /// In certain `unsafe` code, lifetime elision combined with
    /// inconsistent lifetime syntax may result in unsound code.
    pub MISMATCHED_LIFETIME_SYNTAXES,
    Warn,
    "detects when an elided lifetime uses different syntax between arguments and return values"
}

declare_lint_pass!(LifetimeStyle => [MISMATCHED_LIFETIME_SYNTAXES]);

impl<'tcx> LateLintPass<'tcx> for LifetimeStyle {
    #[instrument(skip_all)]
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        _: hir::intravisit::FnKind<'tcx>,
        fd: &'tcx hir::FnDecl<'tcx>,
        _: &'tcx hir::Body<'tcx>,
        _: rustc_span::Span,
        _: rustc_span::def_id::LocalDefId,
    ) {
        let mut input_map = Default::default();
        let mut output_map = Default::default();

        for input in fd.inputs {
            LifetimeInfoCollector::collect(input, &mut input_map);
        }

        if let hir::FnRetTy::Return(output) = fd.output {
            LifetimeInfoCollector::collect(output, &mut output_map);
        }

        report_mismatches(cx, &input_map, &output_map);
    }
}

#[instrument(skip_all)]
fn report_mismatches<'tcx>(
    cx: &LateContext<'tcx>,
    inputs: &LifetimeInfoMap<'tcx>,
    outputs: &LifetimeInfoMap<'tcx>,
) {
    for (resolved_lifetime, output_info) in outputs {
        if let Some(input_info) = inputs.get(resolved_lifetime) {
            let relevant_lifetimes = input_info.iter().chain(output_info);

            // Categorize lifetimes into source/syntax buckets
            let mut hidden = Bucket::default();
            let mut anonymous = Bucket::default();
            let mut named = Bucket::default();

            for info in relevant_lifetimes {
                use LifetimeSource::*;
                use LifetimeSyntax::*;

                let bucket = match info.lifetime.syntax {
                    Hidden => &mut hidden,
                    Anonymous => &mut anonymous,
                    Named => &mut named,
                };

                match info.lifetime.source {
                    Reference | OutlivesBound | PreciseCapturing => bucket.n_ref += 1,
                    Path { .. } => bucket.n_path += 1,
                    Other => {}
                }

                bucket.members.push(info);
            }

            // Check if syntaxes are consistent

            let syntax_counts = (
                hidden.n_ref,
                anonymous.n_ref,
                named.n_ref,
                hidden.n_path,
                anonymous.n_path,
                named.n_path,
            );

            match syntax_counts {
                // The lifetimes are all one syntax
                (_, 0, 0, _, 0, 0) | (0, _, 0, 0, _, 0) | (0, 0, _, 0, 0, _) => continue,

                // Hidden references, anonymous references, and anonymous paths can call be mixed.
                (_, _, 0, 0, _, 0) => continue,

                _ => (),
            }

            let inputs = input_info.iter().map(|info| info.reporting_span()).collect();
            let outputs = output_info.iter().map(|info| info.reporting_span()).collect();

            // There can only ever be zero or one named lifetime
            // for a given lifetime resolution.
            let named_lifetime = named.members.first();

            let named_suggestion = named_lifetime.map(|info| {
                build_mismatch_suggestion(info.lifetime_name(), &[&hidden, &anonymous])
            });

            let is_named_static = named_lifetime.is_some_and(|info| info.is_static());

            let should_suggest_hidden = !hidden.members.is_empty() && !is_named_static;

            // FIXME: remove comma and space from paths with multiple generics
            // FIXME: remove angle brackets from paths when no more generics
            // FIXME: remove space after lifetime from references
            // FIXME: remove lifetime from function declaration
            let hidden_suggestion = should_suggest_hidden.then(|| {
                let suggestions = [&anonymous, &named]
                    .into_iter()
                    .flat_map(|b| &b.members)
                    .map(|i| i.suggestion("'dummy").0)
                    .collect();

                lints::MismatchedLifetimeSyntaxesSuggestion::Hidden {
                    suggestions,
                    tool_only: false,
                }
            });

            let should_suggest_anonymous = !anonymous.members.is_empty() && !is_named_static;

            let anonymous_suggestion = should_suggest_anonymous
                .then(|| build_mismatch_suggestion("'_", &[&hidden, &named]));

            let lifetime_name =
                named_lifetime.map(|info| info.lifetime_name()).unwrap_or("'_").to_owned();

            // We can produce a number of suggestions which may overwhelm
            // the user. Instead, we order the suggestions based on Rust
            // idioms. The "best" choice is shown to the user and the
            // remaining choices are shown to tools only.
            let mut suggestions = Vec::new();
            suggestions.extend(named_suggestion);
            suggestions.extend(anonymous_suggestion);
            suggestions.extend(hidden_suggestion);

            cx.emit_span_lint(
                MISMATCHED_LIFETIME_SYNTAXES,
                Vec::clone(&inputs),
                lints::MismatchedLifetimeSyntaxes { lifetime_name, inputs, outputs, suggestions },
            );
        }
    }
}

#[derive(Default)]
struct Bucket<'a, 'tcx> {
    members: Vec<&'a Info<'tcx>>,
    n_ref: usize,
    n_path: usize,
}

fn build_mismatch_suggestion(
    lifetime_name: &str,
    buckets: &[&Bucket<'_, '_>],
) -> lints::MismatchedLifetimeSyntaxesSuggestion {
    let lifetime_name = lifetime_name.to_owned();

    let suggestions = buckets
        .iter()
        .flat_map(|b| &b.members)
        .map(|info| info.suggestion(&lifetime_name))
        .collect();

    lints::MismatchedLifetimeSyntaxesSuggestion::Named {
        lifetime_name,
        suggestions,
        tool_only: false,
    }
}

struct Info<'tcx> {
    type_span: Span,
    lifetime: &'tcx hir::Lifetime,
}

impl<'tcx> Info<'tcx> {
    fn lifetime_name(&self) -> &str {
        self.lifetime.ident.as_str()
    }

    fn is_static(&self) -> bool {
        self.lifetime.is_static()
    }

    /// When reporting a lifetime that is hidden, we expand the span
    /// to include the type. Otherwise we end up pointing at nothing,
    /// which is a bit confusing.
    fn reporting_span(&self) -> Span {
        if self.lifetime.is_syntactically_hidden() {
            self.type_span
        } else {
            self.lifetime.ident.span
        }
    }

    fn suggestion(&self, lifetime_name: &str) -> (Span, String) {
        self.lifetime.suggestion(lifetime_name)
    }
}

type LifetimeInfoMap<'tcx> = FxIndexMap<&'tcx hir::LifetimeName, Vec<Info<'tcx>>>;

struct LifetimeInfoCollector<'a, 'tcx> {
    type_span: Span,
    map: &'a mut LifetimeInfoMap<'tcx>,
}

impl<'a, 'tcx> LifetimeInfoCollector<'a, 'tcx> {
    fn collect(ty: &'tcx hir::Ty<'tcx>, map: &'a mut LifetimeInfoMap<'tcx>) {
        let mut this = Self { type_span: ty.span, map };

        intravisit::walk_unambig_ty(&mut this, ty);
    }
}

impl<'a, 'tcx> Visitor<'tcx> for LifetimeInfoCollector<'a, 'tcx> {
    #[instrument(skip(self))]
    fn visit_lifetime(&mut self, lifetime: &'tcx hir::Lifetime) {
        let type_span = self.type_span;

        let info = Info { type_span, lifetime };

        self.map.entry(&lifetime.res).or_default().push(info);
    }

    #[instrument(skip(self))]
    fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx, hir::AmbigArg>) -> Self::Result {
        let old_type_span = self.type_span;

        self.type_span = ty.span;

        intravisit::walk_ty(self, ty);

        self.type_span = old_type_span;
    }
}
