//! Disjunctive Normal Form construction.
//!
//! Algorithm from <https://www.cs.drexel.edu/~jjohnson/2015-16/fall/CS270/Lectures/3/dnf.pdf>,
//! which would have been much easier to read if it used pattern matching. It's also missing the
//! entire "distribute ANDs over ORs" part, which is not trivial. Oh well.
//!
//! This is currently both messy and inefficient. Feel free to improve, there are unit tests.

use std::fmt::{self, Write};

use rustc_hash::FxHashSet;

use crate::{CfgAtom, CfgDiff, CfgExpr, CfgOptions, InactiveReason};

/// A `#[cfg]` directive in Disjunctive Normal Form (DNF).
pub struct DnfExpr {
    conjunctions: Vec<Conjunction>,
}

struct Conjunction {
    literals: Vec<Literal>,
}

struct Literal {
    negate: bool,
    var: Option<CfgAtom>, // None = Invalid
}

impl DnfExpr {
    pub fn new(expr: &CfgExpr) -> Self {
        let builder = Builder { expr: DnfExpr { conjunctions: Vec::new() } };

        builder.lower(expr)
    }

    /// Computes a list of present or absent atoms in `opts` that cause this expression to evaluate
    /// to `false`.
    ///
    /// Note that flipping a subset of these atoms might be sufficient to make the whole expression
    /// evaluate to `true`. For that, see `compute_enable_hints`.
    ///
    /// Returns `None` when `self` is already true, or contains errors.
    pub fn why_inactive(&self, opts: &CfgOptions) -> Option<InactiveReason> {
        let mut res = InactiveReason { enabled: Vec::new(), disabled: Vec::new() };

        for conj in &self.conjunctions {
            let mut conj_is_true = true;
            for lit in &conj.literals {
                let atom = lit.var.as_ref()?;
                let enabled = opts.enabled.contains(atom);
                if lit.negate == enabled {
                    // Literal is false, but needs to be true for this conjunction.
                    conj_is_true = false;

                    if enabled {
                        res.enabled.push(atom.clone());
                    } else {
                        res.disabled.push(atom.clone());
                    }
                }
            }

            if conj_is_true {
                // This expression is not actually inactive.
                return None;
            }
        }

        res.enabled.sort_unstable();
        res.enabled.dedup();
        res.disabled.sort_unstable();
        res.disabled.dedup();
        Some(res)
    }

    /// Returns `CfgDiff` objects that would enable this directive if applied to `opts`.
    pub fn compute_enable_hints<'a>(
        &'a self,
        opts: &'a CfgOptions,
    ) -> impl Iterator<Item = CfgDiff> + 'a {
        // A cfg is enabled if any of `self.conjunctions` evaluate to `true`.

        self.conjunctions.iter().filter_map(move |conj| {
            let mut enable = FxHashSet::default();
            let mut disable = FxHashSet::default();
            for lit in &conj.literals {
                let atom = lit.var.as_ref()?;
                let enabled = opts.enabled.contains(atom);
                if lit.negate && enabled {
                    disable.insert(atom.clone());
                }
                if !lit.negate && !enabled {
                    enable.insert(atom.clone());
                }
            }

            // Check that this actually makes `conj` true.
            for lit in &conj.literals {
                let atom = lit.var.as_ref()?;
                let enabled = enable.contains(atom)
                    || (opts.enabled.contains(atom) && !disable.contains(atom));
                if enabled == lit.negate {
                    return None;
                }
            }

            if enable.is_empty() && disable.is_empty() {
                return None;
            }

            let mut diff = CfgDiff {
                enable: enable.into_iter().collect(),
                disable: disable.into_iter().collect(),
            };

            // Undo the FxHashMap randomization for consistent output.
            diff.enable.sort_unstable();
            diff.disable.sort_unstable();

            Some(diff)
        })
    }
}

impl fmt::Display for DnfExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.conjunctions.len() != 1 {
            f.write_str("any(")?;
        }
        for (i, conj) in self.conjunctions.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }

            conj.fmt(f)?;
        }
        if self.conjunctions.len() != 1 {
            f.write_char(')')?;
        }

        Ok(())
    }
}

impl Conjunction {
    fn new(parts: Box<[CfgExpr]>) -> Self {
        let mut literals = Vec::new();
        for part in parts.into_vec() {
            match part {
                CfgExpr::Invalid | CfgExpr::Atom(_) | CfgExpr::Not(_) => {
                    literals.push(Literal::new(part));
                }
                CfgExpr::All(conj) => {
                    // Flatten.
                    literals.extend(Conjunction::new(conj).literals);
                }
                CfgExpr::Any(_) => unreachable!("disjunction in conjunction"),
            }
        }

        Self { literals }
    }
}

impl fmt::Display for Conjunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.literals.len() != 1 {
            f.write_str("all(")?;
        }
        for (i, lit) in self.literals.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }

            lit.fmt(f)?;
        }
        if self.literals.len() != 1 {
            f.write_str(")")?;
        }

        Ok(())
    }
}

impl Literal {
    fn new(expr: CfgExpr) -> Self {
        match expr {
            CfgExpr::Invalid => Self { negate: false, var: None },
            CfgExpr::Atom(atom) => Self { negate: false, var: Some(atom) },
            CfgExpr::Not(expr) => match *expr {
                CfgExpr::Invalid => Self { negate: true, var: None },
                CfgExpr::Atom(atom) => Self { negate: true, var: Some(atom) },
                _ => unreachable!("non-atom {:?}", expr),
            },
            CfgExpr::Any(_) | CfgExpr::All(_) => unreachable!("non-literal {:?}", expr),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.negate {
            write!(f, "not(")?;
        }

        match &self.var {
            Some(var) => var.fmt(f)?,
            None => f.write_str("<invalid>")?,
        }

        if self.negate {
            f.write_char(')')?;
        }

        Ok(())
    }
}

struct Builder {
    expr: DnfExpr,
}

impl Builder {
    fn lower(mut self, expr: &CfgExpr) -> DnfExpr {
        let expr = make_nnf(expr);
        let expr = make_dnf(expr);

        match expr {
            CfgExpr::Invalid | CfgExpr::Atom(_) | CfgExpr::Not(_) => {
                self.expr.conjunctions.push(Conjunction::new(Box::new([expr])));
            }
            CfgExpr::All(conj) => {
                self.expr.conjunctions.push(Conjunction::new(conj));
            }
            CfgExpr::Any(disj) => {
                let mut disj = disj.into_vec();
                disj.reverse();
                while let Some(conj) = disj.pop() {
                    match conj {
                        CfgExpr::Invalid | CfgExpr::Atom(_) | CfgExpr::All(_) | CfgExpr::Not(_) => {
                            self.expr.conjunctions.push(Conjunction::new(Box::new([conj])));
                        }
                        CfgExpr::Any(inner_disj) => {
                            // Flatten.
                            disj.extend(inner_disj.into_vec().into_iter().rev());
                        }
                    }
                }
            }
        }

        self.expr
    }
}

fn make_dnf(expr: CfgExpr) -> CfgExpr {
    match expr {
        CfgExpr::Invalid | CfgExpr::Atom(_) | CfgExpr::Not(_) => expr,
        CfgExpr::Any(e) => flatten(CfgExpr::Any(e.into_vec().into_iter().map(make_dnf).collect())),
        CfgExpr::All(e) => {
            let e = e.into_vec().into_iter().map(make_dnf).collect::<Vec<_>>();

            flatten(CfgExpr::Any(distribute_conj(&e).into_boxed_slice()))
        }
    }
}

/// Turns a conjunction of expressions into a disjunction of expressions.
fn distribute_conj(conj: &[CfgExpr]) -> Vec<CfgExpr> {
    fn go(out: &mut Vec<CfgExpr>, with: &mut Vec<CfgExpr>, rest: &[CfgExpr]) {
        match rest {
            [head, tail @ ..] => match head {
                CfgExpr::Any(disj) => {
                    for part in disj.iter() {
                        with.push(part.clone());
                        go(out, with, tail);
                        with.pop();
                    }
                }
                _ => {
                    with.push(head.clone());
                    go(out, with, tail);
                    with.pop();
                }
            },
            _ => {
                // Turn accumulated parts into a new conjunction.
                out.push(CfgExpr::All(with.clone().into_boxed_slice()));
            }
        }
    }

    let mut out = Vec::new(); // contains only `all()`
    let mut with = Vec::new();

    go(&mut out, &mut with, conj);

    out
}

fn make_nnf(expr: &CfgExpr) -> CfgExpr {
    match expr {
        CfgExpr::Invalid | CfgExpr::Atom(_) => expr.clone(),
        CfgExpr::Any(expr) => CfgExpr::Any(expr.iter().map(make_nnf).collect()),
        CfgExpr::All(expr) => CfgExpr::All(expr.iter().map(make_nnf).collect()),
        CfgExpr::Not(operand) => make_nnf_neg(operand),
    }
}

fn make_nnf_neg(operand: &CfgExpr) -> CfgExpr {
    match operand {
        // Original negated expr
        CfgExpr::Invalid => CfgExpr::Not(Box::new(CfgExpr::Invalid)), // Original negated expr
        // Original negated expr
        CfgExpr::Atom(atom) => CfgExpr::Not(Box::new(CfgExpr::Atom(atom.clone()))),
        // Remove double negation.
        CfgExpr::Not(expr) => make_nnf(expr),
        // Convert negated conjunction/disjunction using DeMorgan's Law.
        CfgExpr::Any(inner) => CfgExpr::All(inner.iter().map(make_nnf_neg).collect()),
        // Convert negated conjunction/disjunction using DeMorgan's Law.
        CfgExpr::All(inner) => CfgExpr::Any(inner.iter().map(make_nnf_neg).collect()),
    }
}

/// Collapses nested `any()` and `all()` predicates.
fn flatten(expr: CfgExpr) -> CfgExpr {
    match expr {
        CfgExpr::All(inner) => CfgExpr::All(
            inner
                .iter()
                .flat_map(|e| match e {
                    CfgExpr::All(inner) => inner.as_ref(),
                    _ => std::slice::from_ref(e),
                })
                .cloned()
                .collect(),
        ),
        CfgExpr::Any(inner) => CfgExpr::Any(
            inner
                .iter()
                .flat_map(|e| match e {
                    CfgExpr::Any(inner) => inner.as_ref(),
                    _ => std::slice::from_ref(e),
                })
                .cloned()
                .collect(),
        ),
        _ => expr,
    }
}
