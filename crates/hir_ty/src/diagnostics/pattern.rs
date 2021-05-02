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

    #[test]
    fn c_enum() {
        check_diagnostics(
            r#"
enum E { A, B }
fn main(v: E) {
    match v { E::A | E::B => {} }
    match v { _           => {} }
    match v { E::A        => {} }
        //^ Missing match arm
    match v {                   }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn enum_() {
        check_diagnostics(
            r#"
struct A; struct B;
enum E { Tuple(A, B), Struct{ a: A, b: B } }
fn main(v: E) {
    match v {
        E::Tuple(a, b)    => {}
        E::Struct{ a, b } => {}
    }
    match v {
        E::Tuple(_, _) => {}
        E::Struct{..}  => {}
    }
    match v {
        E::Tuple(..) => {}
        _ => {}
    }
    match v { E::Tuple(..) => {} }
        //^ Missing match arm
    match v { }
        //^ Missing match arm
}
"#,
        );
    }

    #[test]
    fn boolean() {
        check_diagnostics(
            r#"
fn main() {
    match true {
        true  => {}
        false => {}
    }
    match true {
        true | false => {}
    }
    match true {
        true => {}
        _ => {}
    }
    match true {}
        //^^^^ Missing match arm
    match true { true => {} }
        //^^^^ Missing match arm

}
"#,
        );
    }
}
