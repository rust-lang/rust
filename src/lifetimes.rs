use rustc_front::hir::*;
use reexport::*;
use rustc::lint::*;
use syntax::codemap::Span;
use rustc_front::visit::{Visitor, walk_ty, walk_ty_param_bound};
use rustc::middle::def::Def::{DefTy, DefTrait};
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
}

impl LateLintPass for LifetimePass {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        if let ItemFn(ref decl, _, _, _, ref generics, _) = item.node {
            check_fn_inner(cx, decl, None, &generics, item.span);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext, item: &ImplItem) {
        if let MethodImplItem(ref sig, _) = item.node {
            check_fn_inner(cx, &sig.decl, Some(&sig.explicit_self),
                           &sig.generics, item.span);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext, item: &TraitItem) {
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

fn check_fn_inner(cx: &LateContext, decl: &FnDecl, slf: Option<&ExplicitSelf>,
                  generics: &Generics, span: Span) {
    if in_external_macro(cx, span) || has_where_lifetimes(cx, &generics.where_clause) {
        return;
    }
    if could_use_elision(cx, decl, slf, &generics.lifetimes) {
        span_lint(cx, NEEDLESS_LIFETIMES, span,
                  "explicit lifetimes given in parameter types where they could be elided");
    }
}

fn could_use_elision(cx: &LateContext, func: &FnDecl, slf: Option<&ExplicitSelf>,
                     named_lts: &[LifetimeDef]) -> bool {
    // There are two scenarios where elision works:
    // * no output references, all input references have different LT
    // * output references, exactly one input reference with same LT
    // All lifetimes must be unnamed, 'static or defined without bounds on the
    // level of the current item.

    // check named LTs
    let allowed_lts = allowed_lts_from(named_lts);

    // these will collect all the lifetimes for references in arg/return types
    let mut input_visitor = RefVisitor::new(cx);
    let mut output_visitor = RefVisitor::new(cx);

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
        input_visitor.visit_ty(&arg.ty);
    }
    // extract lifetimes in output type
    if let Return(ref ty) = func.output {
        output_visitor.visit_ty(ty);
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

fn allowed_lts_from(named_lts: &[LifetimeDef]) -> HashSet<RefLt> {
    let mut allowed_lts = HashSet::new();
    for lt in named_lts {
        if lt.bounds.is_empty() {
            allowed_lts.insert(Named(lt.lifetime.name));
        }
    }
    allowed_lts.insert(Unnamed);
    allowed_lts.insert(Static);
    allowed_lts
}

/// Number of unique lifetimes in the given vector.
fn unique_lifetimes(lts: &[RefLt]) -> usize {
    lts.iter().collect::<HashSet<_>>().len()
}

/// A visitor usable for rustc_front::visit::walk_ty().
struct RefVisitor<'v, 't: 'v> {
    cx: &'v LateContext<'v, 't>, // context reference
    lts: Vec<RefLt>
}

impl <'v, 't> RefVisitor<'v, 't>  {
    fn new(cx: &'v LateContext<'v, 't>) -> RefVisitor<'v, 't> {
        RefVisitor { cx: cx, lts: Vec::new() }
    }

    fn record(&mut self, lifetime: &Option<Lifetime>) {
        if let &Some(ref lt) = lifetime {
            if lt.name.as_str() == "'static" {
                self.lts.push(Static);
            } else {
                self.lts.push(Named(lt.name));
            }
        } else {
            self.lts.push(Unnamed);
        }
    }

    fn into_vec(self) -> Vec<RefLt> {
        self.lts
    }

    fn collect_anonymous_lifetimes(&mut self, path: &Path, ty: &Ty) {
        let last_path_segment = path.segments.last().map(|s| &s.parameters);
        if let Some(&AngleBracketedParameters(ref params)) = last_path_segment {
            if params.lifetimes.is_empty() {
                let def = self.cx.tcx.def_map.borrow().get(&ty.id).map(|r| r.full_def());
                match def {
                    Some(DefTy(def_id, _)) => {
                        if let Some(ty_def) = self.cx.tcx.adt_defs.borrow().get(&def_id) {
                            let scheme = ty_def.type_scheme(self.cx.tcx);
                            for _ in scheme.generics.regions.as_slice() {
                                self.record(&None);
                            }
                        }
                    }
                    Some(DefTrait(def_id)) => {
                        let trait_def = self.cx.tcx.trait_defs.borrow()[&def_id];
                        for _ in &trait_def.generics.regions {
                            self.record(&None);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

impl<'v, 't> Visitor<'v> for RefVisitor<'v, 't> {

    // for lifetimes as parameters of generics
    fn visit_lifetime(&mut self, lifetime: &'v Lifetime) {
        self.record(&Some(*lifetime));
    }

    fn visit_ty(&mut self, ty: &'v Ty) {
        match ty.node {
            TyRptr(None, _) => {
                self.record(&None);
            }
            TyPath(_, ref path) => {
                self.collect_anonymous_lifetimes(path, ty);
            }
            _ => {}
        }
        walk_ty(self, ty);
    }
}

/// Are any lifetimes mentioned in the `where` clause? If yes, we don't try to
/// reason about elision.
fn has_where_lifetimes(cx: &LateContext, where_clause: &WhereClause) -> bool {
    for predicate in &where_clause.predicates {
        match *predicate {
            WherePredicate::RegionPredicate(..) => return true,
            WherePredicate::BoundPredicate(ref pred) => {
                // a predicate like F: Trait or F: for<'a> Trait<'a>
                let mut visitor = RefVisitor::new(cx);
                // walk the type F, it may not contain LT refs
                walk_ty(&mut visitor, &pred.bounded_ty);
                if !visitor.lts.is_empty() { return true; }
                // if the bounds define new lifetimes, they are fine to occur
                let allowed_lts = allowed_lts_from(&pred.bound_lifetimes);
                // now walk the bounds
                for bound in pred.bounds.iter() {
                    walk_ty_param_bound(&mut visitor, bound);
                }
                // and check that all lifetimes are allowed
                for lt in visitor.into_vec() {
                    if !allowed_lts.contains(&lt) {
                        return true;
                    }
                }
            }
            WherePredicate::EqPredicate(ref pred) => {
                let mut visitor = RefVisitor::new(cx);
                walk_ty(&mut visitor, &pred.ty);
                if !visitor.lts.is_empty() { return true; }
            }
        }
    }
    false
}
