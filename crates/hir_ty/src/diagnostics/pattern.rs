#![deny(elided_lifetimes_in_paths)]
#![allow(unused)] // todo remove

mod deconstruct_pat;
pub mod usefulness;

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    use super::*;

    #[test]
    fn unit() {
        check_diagnostics(
            r#"
fn main() {
    match () { () => {} }
    match () {  _ => {} }
    match () {          }
        //^^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn tuple_of_units() {
        check_diagnostics(
            r#"
fn main() {
    match ((), ()) { ((), ()) => {} }
    match ((), ()) {  ((), _) => {} }
    match ((), ()) {   (_, _) => {} }
    match ((), ()) {        _ => {} }
    match ((), ()) {                }
        //^^^^^^^^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn tuple_with_ellipsis() {
        // TODO: test non-exhaustive match with ellipsis in the middle
        // of a pattern, check reported witness
        check_diagnostics(
            r#"
struct A; struct B;
fn main(v: (A, (), B)) {
    match v { (A, ..)    => {} }
    match v { (.., B)    => {} }
    match v { (A, .., B) => {} }
    match v { (..)       => {} }
    match v {                  }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn strukt() {
        check_diagnostics(
            r#"
struct A; struct B;
struct S { a: A, b: B}
fn main(v: S) {
    match v { S { a, b }       => {} }
    match v { S { a: _, b: _ } => {} }
    match v { S { .. }         => {} }
    match v { _                => {} }
    match v {                        }
        //^ Missing match arm
}
"#,
        );
    }
}
