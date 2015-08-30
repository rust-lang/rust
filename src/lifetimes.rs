use syntax::ast::*;
use rustc::lint::*;
use syntax::codemap::Span;
use syntax::visit::{Visitor, walk_ty};
use std::collections::HashSet;

use utils::{in_external_macro, span_lint};

declare_lint!(pub NEEDLESS_LIFETIMES, Warn,
              "using explicit lifetimes for references in function arguments when elision rules \
               would allow omitting them");

#[derive(Copy,Clone)]
pub struct LifetimePass;

impl LintPass for LifetimePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEEDLESS_LIFETIMES)
    }

    fn check_item(&mut self, cx: &Context, item: &Item) {
        if let ItemFn(ref decl, _, _, _, ref generics, _) = item.node {
            check_fn_inner(cx, decl, None, &generics, item.span);
        }
    }

    fn check_impl_item(&mut self, cx: &Context, item: &ImplItem) {
        if let MethodImplItem(ref sig, _) = item.node {
            check_fn_inner(cx, &sig.decl, Some(&sig.explicit_self),
                           &sig.generics, item.span);
        }
    }

    fn check_trait_item(&mut self, cx: &Context, item: &TraitItem) {
        if let MethodTraitItem(ref sig, _) = item.node {
            check_fn_inner(cx, &sig.decl, Some(&sig.explicit_self),
                           &sig.generics, item.span);
        }
    }
}

/// The lifetime of a &-reference.
#[derive(PartialEq, Eq, Hash, Debug)]
enum RefLt {
    Unnamed,
    Static,
    Named(Name),
}
use self::RefLt::*;

fn check_fn_inner(cx: &Context, decl: &FnDecl, slf: Option<&ExplicitSelf>,
                  generics: &Generics, span: Span) {
    if in_external_macro(cx, span) || has_where_lifetimes(&generics.where_clause) {
        return;
    }
    if could_use_elision(decl, slf, &generics.lifetimes) {
        span_lint(cx, NEEDLESS_LIFETIMES, span,
                  "explicit lifetimes given in parameter types where they could be elided");
    }
}

fn could_use_elision(func: &FnDecl, slf: Option<&ExplicitSelf>,
                     named_lts: &[LifetimeDef]) -> bool {
    // There are two scenarios where elision works:
    // * no output references, all input references have different LT
    // * output references, exactly one input reference with same LT
    // All lifetimes must be unnamed, 'static or defined without bounds on the
    // level of the current item.

    // check named LTs
    let mut allowed_lts = HashSet::new();
    for lt in named_lts {
        if lt.bounds.is_empty() {
            allowed_lts.insert(Named(lt.lifetime.name));
        }
    }
    allowed_lts.insert(Unnamed);
    allowed_lts.insert(Static);

    // these will collect all the lifetimes for references in arg/return types
    let mut input_visitor = RefVisitor(Vec::new());
    let mut output_visitor = RefVisitor(Vec::new());

    // extract lifetime in "self" argument for methods (there is a "self" argument
    // in func.inputs, but its type is TyInfer)
    if let Some(slf) = slf {
        match slf.node {
            SelfRegion(ref opt_lt, _, _) => input_visitor.record(opt_lt),
            SelfExplicit(ref ty, _) => walk_ty(&mut input_visitor, ty),
            _ => { }
        }
    }
    // extract lifetimes in input argument types
    for arg in &func.inputs {
        walk_ty(&mut input_visitor, &arg.ty);
    }
    // extract lifetimes in output type
    if let Return(ref ty) = func.output {
        walk_ty(&mut output_visitor, ty);
    }

    let input_lts = input_visitor.into_vec();
    let output_lts = output_visitor.into_vec();

    // check for lifetimes from higher scopes
    for lt in input_lts.iter().chain(output_lts.iter()) {
        if !allowed_lts.contains(lt) {
            return false;
        }
    }

    // no input lifetimes? easy case!
    if input_lts.is_empty() {
        return false;
    } else if output_lts.is_empty() {
        // no output lifetimes, check distinctness of input lifetimes

        // only unnamed and static, ok
        if input_lts.iter().all(|lt| *lt == Unnamed || *lt == Static) {
            return false;
        }
        // we have no output reference, so we only need all distinct lifetimes
        if input_lts.len() == unique_lifetimes(&input_lts) {
            return true;
        }
    } else {
        // we have output references, so we need one input reference,
        // and all output lifetimes must be the same
        if unique_lifetimes(&output_lts) > 1 {
            return false;
        }
        if input_lts.len() == 1 {
            match (&input_lts[0], &output_lts[0]) {
                (&Named(n1), &Named(n2)) if n1 == n2 => { return true; }
                (&Named(_), &Unnamed) => { return true; }
                (&Unnamed, &Named(_)) => { return true; }
                _ => { } // already elided, different named lifetimes
                         // or something static going on
            }
        }
    }
    false
}

/// Number of unique lifetimes in the given vector.
fn unique_lifetimes(lts: &[RefLt]) -> usize {
    lts.iter().collect::<HashSet<_>>().len()
}

/// A visitor usable for syntax::visit::walk_ty().
struct RefVisitor(Vec<RefLt>);

impl RefVisitor {
    fn record(&mut self, lifetime: &Option<Lifetime>) {
        if let &Some(ref lt) = lifetime {
            if lt.name == "'static" {
                self.0.push(Static);
            } else {
                self.0.push(Named(lt.name));
            }
        } else {
            self.0.push(Unnamed);
        }
    }

    fn into_vec(self) -> Vec<RefLt> {
        self.0
    }
}

impl<'v> Visitor<'v> for RefVisitor {
    // for lifetimes of references
    fn visit_opt_lifetime_ref(&mut self, _: Span, lifetime: &'v Option<Lifetime>) {
        self.record(lifetime);
    }

    // for lifetimes as parameters of generics
    fn visit_lifetime_ref(&mut self, lifetime: &'v Lifetime) {
        self.record(&Some(*lifetime));
    }

    // for lifetime bounds; the default impl calls visit_lifetime_ref
    fn visit_lifetime_bound(&mut self, _: &'v Lifetime) { }
}

/// Are any lifetimes mentioned in the `where` clause? If yes, we don't try to
/// reason about elision.
fn has_where_lifetimes(where_clause: &WhereClause) -> bool {
    let mut where_visitor = RefVisitor(Vec::new());
    for predicate in &where_clause.predicates {
        match *predicate {
            WherePredicate::RegionPredicate(..) => return true,
            WherePredicate::BoundPredicate(ref pred) => {
                walk_ty(&mut where_visitor, &pred.bounded_ty);
            }
            WherePredicate::EqPredicate(ref pred) => {
                walk_ty(&mut where_visitor, &pred.ty);
            }
        }
    }
    !where_visitor.into_vec().is_empty()
}
