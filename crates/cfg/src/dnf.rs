//! Disjunctive Normal Form construction.
//!
//! Algorithm from <https://www.cs.drexel.edu/~jjohnson/2015-16/fall/CS270/Lectures/3/dnf.pdf>,
//! which would have been much easier to read if it used pattern matching. It's also missing the
//! entire "distribute ANDs over ORs" part, which is not trivial. Oh well.
//!
//! This is currently both messy and inefficient. Feel free to improve, there are unit tests.

use std::fmt;

use rustc_hash::FxHashSet;

use crate::{CfgAtom, CfgDiff, CfgExpr, CfgOptions, InactiveReason};

/// A `#[cfg]` directive in Disjunctive Normal Form (DNF).
pub struct DnfExpr {
    conjunctions: Vec<Conjunction>,
}

impl DnfExpr {
    pub fn new(expr: CfgExpr) -> Self {
        let builder = Builder { expr: DnfExpr { conjunctions: Vec::new() } };

        builder.lower(expr.clone())
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
            write!(f, "any(")?;
        }
        for (i, conj) in self.conjunctions.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }

            write!(f, "{}", conj)?;
        }
        if self.conjunctions.len() != 1 {
            write!(f, ")")?;
        }

        Ok(())
    }
}

struct Conjunction {
    literals: Vec<Literal>,
}

impl Conjunction {
    fn new(parts: Vec<CfgExpr>) -> Self {
        let mut literals = Vec::new();
        for part in parts {
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
            write!(f, "all(")?;
        }
        for (i, lit) in self.literals.iter().enumerate() {
            if i != 0 {
                f.write_str(", ")?;
            }

            write!(f, "{}", lit)?;
        }
        if self.literals.len() != 1 {
            write!(f, ")")?;
        }

        Ok(())
    }
}

struct Literal {
    negate: bool,
    var: Option<CfgAtom>, // None = Invalid
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
            Some(var) => write!(f, "{}", var)?,
            None => f.write_str("<invalid>")?,
        }

        if self.negate {
            write!(f, ")")?;
        }

        Ok(())
    }
}

struct Builder {
    expr: DnfExpr,
}

impl Builder {
    fn lower(mut self, expr: CfgExpr) -> DnfExpr {
        let expr = make_nnf(expr);
        let expr = make_dnf(expr);

        match expr {
            CfgExpr::Invalid | CfgExpr::Atom(_) | CfgExpr::Not(_) => {
                self.expr.conjunctions.push(Conjunction::new(vec![expr]));
            }
            CfgExpr::All(conj) => {
                self.expr.conjunctions.push(Conjunction::new(conj));
            }
            CfgExpr::Any(mut disj) => {
                disj.reverse();
                while let Some(conj) = disj.pop() {
                    match conj {
                        CfgExpr::Invalid | CfgExpr::Atom(_) | CfgExpr::All(_) | CfgExpr::Not(_) => {
                            self.expr.conjunctions.push(Conjunction::new(vec![conj]));
                        }
                        CfgExpr::Any(inner_disj) => {
                            // Flatten.
                            disj.extend(inner_disj.into_iter().rev());
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
        CfgExpr::Any(e) => CfgExpr::Any(e.into_iter().map(|expr| make_dnf(expr)).collect()),
        CfgExpr::All(e) => {
            let e = e.into_iter().map(|expr| make_nnf(expr)).collect::<Vec<_>>();

            CfgExpr::Any(distribute_conj(&e))
        }
    }
}

/// Turns a conjunction of expressions into a disjunction of expressions.
fn distribute_conj(conj: &[CfgExpr]) -> Vec<CfgExpr> {
    fn go(out: &mut Vec<CfgExpr>, with: &mut Vec<CfgExpr>, rest: &[CfgExpr]) {
        match rest {
            [head, tail @ ..] => match head {
                CfgExpr::Any(disj) => {
                    for part in disj {
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
                out.push(CfgExpr::All(with.clone()));
            }
        }
    }

    let mut out = Vec::new();
    let mut with = Vec::new();

    go(&mut out, &mut with, conj);

    out
}

fn make_nnf(expr: CfgExpr) -> CfgExpr {
    match expr {
        CfgExpr::Invalid | CfgExpr::Atom(_) => expr,
        CfgExpr::Any(expr) => CfgExpr::Any(expr.into_iter().map(|expr| make_nnf(expr)).collect()),
        CfgExpr::All(expr) => CfgExpr::All(expr.into_iter().map(|expr| make_nnf(expr)).collect()),
        CfgExpr::Not(operand) => match *operand {
            CfgExpr::Invalid | CfgExpr::Atom(_) => CfgExpr::Not(operand.clone()), // Original negated expr
            CfgExpr::Not(expr) => {
                // Remove double negation.
                make_nnf(*expr)
            }
            // Convert negated conjunction/disjunction using DeMorgan's Law.
            CfgExpr::Any(inner) => CfgExpr::All(
                inner.into_iter().map(|expr| make_nnf(CfgExpr::Not(Box::new(expr)))).collect(),
            ),
            CfgExpr::All(inner) => CfgExpr::Any(
                inner.into_iter().map(|expr| make_nnf(CfgExpr::Not(Box::new(expr)))).collect(),
            ),
        },
    }
}

#[cfg(test)]
mod test {
    use expect_test::{expect, Expect};
    use mbe::ast_to_token_tree;
    use syntax::{ast, AstNode};

    use super::*;

    fn check_dnf(input: &str, expect: Expect) {
        let (tt, _) = {
            let source_file = ast::SourceFile::parse(input).ok().unwrap();
            let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
            ast_to_token_tree(&tt).unwrap()
        };
        let cfg = CfgExpr::parse(&tt);
        let actual = format!("#![cfg({})]", DnfExpr::new(cfg));
        expect.assert_eq(&actual);
    }

    fn check_why_inactive(input: &str, opts: &CfgOptions, expect: Expect) {
        let (tt, _) = {
            let source_file = ast::SourceFile::parse(input).ok().unwrap();
            let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
            ast_to_token_tree(&tt).unwrap()
        };
        let cfg = CfgExpr::parse(&tt);
        let dnf = DnfExpr::new(cfg);
        let why_inactive = dnf.why_inactive(opts).unwrap().to_string();
        expect.assert_eq(&why_inactive);
    }

    #[track_caller]
    fn check_enable_hints(input: &str, opts: &CfgOptions, expected_hints: &[&str]) {
        let (tt, _) = {
            let source_file = ast::SourceFile::parse(input).ok().unwrap();
            let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
            ast_to_token_tree(&tt).unwrap()
        };
        let cfg = CfgExpr::parse(&tt);
        let dnf = DnfExpr::new(cfg);
        let hints = dnf.compute_enable_hints(opts).map(|diff| diff.to_string()).collect::<Vec<_>>();
        assert_eq!(hints, expected_hints);
    }

    #[test]
    fn smoke() {
        check_dnf("#![cfg(test)]", expect![[r#"#![cfg(test)]"#]]);
        check_dnf("#![cfg(not(test))]", expect![[r#"#![cfg(not(test))]"#]]);
        check_dnf("#![cfg(not(not(test)))]", expect![[r#"#![cfg(test)]"#]]);

        check_dnf("#![cfg(all(a, b))]", expect![[r#"#![cfg(all(a, b))]"#]]);
        check_dnf("#![cfg(any(a, b))]", expect![[r#"#![cfg(any(a, b))]"#]]);

        check_dnf("#![cfg(not(a))]", expect![[r#"#![cfg(not(a))]"#]]);
    }

    #[test]
    fn distribute() {
        check_dnf("#![cfg(all(any(a, b), c))]", expect![[r#"#![cfg(any(all(a, c), all(b, c)))]"#]]);
        check_dnf("#![cfg(all(c, any(a, b)))]", expect![[r#"#![cfg(any(all(c, a), all(c, b)))]"#]]);
        check_dnf(
            "#![cfg(all(any(a, b), any(c, d)))]",
            expect![[r#"#![cfg(any(all(a, c), all(a, d), all(b, c), all(b, d)))]"#]],
        );

        check_dnf(
            "#![cfg(all(any(a, b, c), any(d, e, f), g))]",
            expect![[
                r#"#![cfg(any(all(a, d, g), all(a, e, g), all(a, f, g), all(b, d, g), all(b, e, g), all(b, f, g), all(c, d, g), all(c, e, g), all(c, f, g)))]"#
            ]],
        );
    }

    #[test]
    fn demorgan() {
        check_dnf("#![cfg(not(all(a, b)))]", expect![[r#"#![cfg(any(not(a), not(b)))]"#]]);
        check_dnf("#![cfg(not(any(a, b)))]", expect![[r#"#![cfg(all(not(a), not(b)))]"#]]);

        check_dnf("#![cfg(not(all(not(a), b)))]", expect![[r#"#![cfg(any(a, not(b)))]"#]]);
        check_dnf("#![cfg(not(any(a, not(b))))]", expect![[r#"#![cfg(all(not(a), b))]"#]]);
    }

    #[test]
    fn nested() {
        check_dnf(
            "#![cfg(all(any(a), not(all(any(b)))))]",
            expect![[r#"#![cfg(all(a, not(b)))]"#]],
        );

        check_dnf("#![cfg(any(any(a, b)))]", expect![[r#"#![cfg(any(a, b))]"#]]);
        check_dnf("#![cfg(not(any(any(a, b))))]", expect![[r#"#![cfg(all(not(a), not(b)))]"#]]);
        check_dnf("#![cfg(all(all(a, b)))]", expect![[r#"#![cfg(all(a, b))]"#]]);
        check_dnf("#![cfg(not(all(all(a, b))))]", expect![[r#"#![cfg(any(not(a), not(b)))]"#]]);
    }

    #[test]
    fn hints() {
        let mut opts = CfgOptions::default();

        check_enable_hints("#![cfg(test)]", &opts, &["enable test"]);
        check_enable_hints("#![cfg(not(test))]", &opts, &[]);

        check_enable_hints("#![cfg(any(a, b))]", &opts, &["enable a", "enable b"]);
        check_enable_hints("#![cfg(any(b, a))]", &opts, &["enable b", "enable a"]);

        check_enable_hints("#![cfg(all(a, b))]", &opts, &["enable a and b"]);

        opts.insert_atom("test".into());

        check_enable_hints("#![cfg(test)]", &opts, &[]);
        check_enable_hints("#![cfg(not(test))]", &opts, &["disable test"]);
    }

    /// Tests that we don't suggest hints for cfgs that express an inconsistent formula.
    #[test]
    fn hints_impossible() {
        let mut opts = CfgOptions::default();

        check_enable_hints("#![cfg(all(test, not(test)))]", &opts, &[]);

        opts.insert_atom("test".into());

        check_enable_hints("#![cfg(all(test, not(test)))]", &opts, &[]);
    }

    #[test]
    fn why_inactive() {
        let mut opts = CfgOptions::default();
        opts.insert_atom("test".into());
        opts.insert_atom("test2".into());

        check_why_inactive("#![cfg(a)]", &opts, expect![["a is disabled"]]);
        check_why_inactive("#![cfg(not(test))]", &opts, expect![["test is enabled"]]);

        check_why_inactive(
            "#![cfg(all(not(test), not(test2)))]",
            &opts,
            expect![["test and test2 are enabled"]],
        );
        check_why_inactive("#![cfg(all(a, b))]", &opts, expect![["a and b are disabled"]]);
        check_why_inactive(
            "#![cfg(all(not(test), a))]",
            &opts,
            expect![["test is enabled and a is disabled"]],
        );
        check_why_inactive(
            "#![cfg(all(not(test), test2, a))]",
            &opts,
            expect![["test is enabled and a is disabled"]],
        );
        check_why_inactive(
            "#![cfg(all(not(test), not(test2), a))]",
            &opts,
            expect![["test and test2 are enabled and a is disabled"]],
        );
    }
}
