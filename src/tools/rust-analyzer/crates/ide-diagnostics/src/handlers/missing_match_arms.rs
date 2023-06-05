use crate::{Diagnostic, DiagnosticsContext};

// Diagnostic: missing-match-arm
//
// This diagnostic is triggered if `match` block is missing one or more match arms.
pub(crate) fn missing_match_arms(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MissingMatchArms,
) -> Diagnostic {
    Diagnostic::new(
        "missing-match-arm",
        format!("missing match arm: {}", d.uncovered_patterns),
        ctx.sema.diagnostics_display_range(d.scrutinee_expr.clone().map(Into::into)).range,
    )
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    fn check_diagnostics_no_bails(ra_fixture: &str) {
        cov_mark::check_count!(validate_match_bailed_out, 0);
        crate::tests::check_diagnostics(ra_fixture)
    }

    #[test]
    fn empty_tuple() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match () { }
        //^^ error: missing match arm: type `()` is non-empty
    match (()) { }
        //^^^^ error: missing match arm: type `()` is non-empty

    match () { _ => (), }
    match () { () => (), }
    match (()) { (()) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_two_empty_tuple() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match ((), ()) { }
        //^^^^^^^^ error: missing match arm: type `((), ())` is non-empty

    match ((), ()) { ((), ()) => (), }
}
"#,
        );
    }

    #[test]
    fn boolean() {
        check_diagnostics_no_bails(
            r#"
fn test_main() {
    match false { }
        //^^^^^ error: missing match arm: type `bool` is non-empty
    match false { true => (), }
        //^^^^^ error: missing match arm: `false` not covered
    match (false, true) {}
        //^^^^^^^^^^^^^ error: missing match arm: type `(bool, bool)` is non-empty
    match (false, true) { (true, true) => (), }
        //^^^^^^^^^^^^^ error: missing match arm: `(false, _)` not covered
    match (false, true) {
        //^^^^^^^^^^^^^ error: missing match arm: `(true, true)` not covered
        (false, true) => (),
        (false, false) => (),
        (true, false) => (),
    }
    match (false, true) { (true, _x) => (), }
        //^^^^^^^^^^^^^ error: missing match arm: `(false, _)` not covered

    match false { true => (), false => (), }
    match (false, true) {
        (false, _) => (),
        (true, false) => (),
        (_, true) => (),
    }
    match (false, true) {
        (true, true) => (),
        (true, false) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true) {
        (true, _x) => (),
        (false, true) => (),
        (false, false) => (),
    }
    match (false, true, false) {
        (false, ..) => (),
        (true, ..) => (),
    }
    match (false, true, false) {
        (.., false) => (),
        (.., true) => (),
    }
    match (false, true, false) { (..) => (), }
}
"#,
        );
    }

    #[test]
    fn tuple_of_tuple_and_bools() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match (false, ((), false)) {}
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: type `(bool, ((), bool))` is non-empty
    match (false, ((), false)) { (true, ((), true)) => (), }
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(false, _)` not covered
    match (false, ((), false)) { (true, _) => (), }
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(false, _)` not covered

    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, ((), true)) => (),
        (false, ((), false)) => (),
    }
    match (false, ((), false)) {
        (true, ((), true)) => (),
        (true, ((), false)) => (),
        (false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn enums() {
        check_diagnostics_no_bails(
            r#"
enum Either { A, B, }

fn main() {
    match Either::A { }
        //^^^^^^^^^ error: missing match arm: `A` and `B` not covered
    match Either::B { Either::A => (), }
        //^^^^^^^^^ error: missing match arm: `B` not covered

    match &Either::B {
        //^^^^^^^^^^ error: missing match arm: `&B` not covered
        Either::A => (),
    }

    match Either::B {
        Either::A => (), Either::B => (),
    }
    match &Either::B {
        Either::A => (), Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_containing_bool() {
        check_diagnostics_no_bails(
            r#"
enum Either { A(bool), B }

fn main() {
    match Either::B { }
        //^^^^^^^^^ error: missing match arm: `A(_)` and `B` not covered
    match Either::B {
        //^^^^^^^^^ error: missing match arm: `A(false)` not covered
        Either::A(true) => (), Either::B => ()
    }

    match Either::B {
        Either::A(true) => (),
        Either::A(false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::B => (),
        _ => (),
    }
    match Either::B {
        Either::A(_) => (),
        Either::B => (),
    }

}
        "#,
        );
    }

    #[test]
    fn enum_different_sizes() {
        check_diagnostics_no_bails(
            r#"
enum Either { A(bool), B(bool, bool) }

fn main() {
    match Either::A(false) {
        //^^^^^^^^^^^^^^^^ error: missing match arm: `B(true, _)` not covered
        Either::A(_) => (),
        Either::B(false, _) => (),
    }

    match Either::A(false) {
        Either::A(_) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
    match Either::A(false) {
        Either::A(true) | Either::A(false) => (),
        Either::B(true, _) => (),
        Either::B(false, _) => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_enum_no_diagnostic() {
        check_diagnostics_no_bails(
            r#"
enum Either { A(bool), B(bool, bool) }
enum Either2 { C, D }

fn main() {
    match (Either::A(false), Either2::C) {
        (Either::A(true), _) | (Either::A(false), _) => (),
        (Either::B(true, _), Either2::C) => (),
        (Either::B(false, _), Either2::C) => (),
        (Either::B(_, _), Either2::D) => (),
    }
}
"#,
        );
    }

    #[test]
    fn or_pattern_no_diagnostic() {
        check_diagnostics_no_bails(
            r#"
enum Either {A, B}

fn main() {
    match (Either::A, Either::B) {
        (Either::A | Either::B, _) => (),
    }
}"#,
        )
    }

    #[test]
    fn mismatched_types() {
        cov_mark::check_count!(validate_match_bailed_out, 4);
        // Match statements with arms that don't match the
        // expression pattern do not fire this diagnostic.
        check_diagnostics(
            r#"
enum Either { A, B }
enum Either2 { C, D }

fn main() {
    match Either::A {
        Either2::C => (),
      //^^^^^^^^^^ error: expected Either, found Either2
        Either2::D => (),
      //^^^^^^^^^^ error: expected Either, found Either2
    }
    match (true, false) {
        (true, false, true) => (),
      //^^^^^^^^^^^^^^^^^^^ error: expected (bool, bool), found (bool, bool, bool)
        (true) => (),
      // ^^^^  error: expected (bool, bool), found bool
    }
    match (true, false) { (true,) => {} }
                        //^^^^^^^ error: expected (bool, bool), found (bool,)
    match (0) { () => () }
              //^^ error: expected i32, found ()
    match Unresolved::Bar { Unresolved::Baz => () }
}
        "#,
        );
    }

    #[test]
    fn mismatched_types_in_or_patterns() {
        cov_mark::check_count!(validate_match_bailed_out, 2);
        check_diagnostics(
            r#"
fn main() {
    match false { true | () => {} }
                       //^^ error: expected bool, found ()
    match (false,) { (true | (),) => {} }
                           //^^ error: expected bool, found ()
}
"#,
        );
    }

    #[test]
    fn malformed_match_arm_tuple_enum_missing_pattern() {
        // We are testing to be sure we don't panic here when the match
        // arm `Either::B` is missing its pattern.
        check_diagnostics_no_bails(
            r#"
enum Either { A, B(u32) }

fn main() {
    match Either::A {
        Either::A => (),
        Either::B() => (),
    }
}
"#,
        );
    }

    #[test]
    fn malformed_match_arm_extra_fields() {
        cov_mark::check_count!(validate_match_bailed_out, 2);
        check_diagnostics(
            r#"
enum A { B(isize, isize), C }
fn main() {
    match A::B(1, 2) {
        A::B(_, _, _) => (),
    }
    match A::B(1, 2) {
        A::C(_) => (),
    }
}
"#,
        );
    }

    #[test]
    fn expr_diverges() {
        cov_mark::check_count!(validate_match_bailed_out, 2);
        check_diagnostics(
            r#"
enum Either { A, B }

fn main() {
    match loop {} {
        Either::A => (),
        Either::B => (),
    }
    match loop {} {
        Either::A => (),
    }
    match loop { break Foo::A } {
        //^^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `B` not covered
        Either::A => (),
    }
    match loop { break Foo::A } {
        Either::A => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn expr_partially_diverges() {
        check_diagnostics_no_bails(
            r#"
enum Either<T> { A(T), B }

fn foo() -> Either<!> { Either::B }
fn main() -> u32 {
    match foo() {
        Either::A(val) => val,
        Either::B => 0,
    }
}
"#,
        );
    }

    #[test]
    fn enum_record() {
        check_diagnostics_no_bails(
            r#"
enum Either { A { foo: bool }, B }

fn main() {
    let a = Either::A { foo: true };
    match a { }
        //^ error: missing match arm: `A { .. }` and `B` not covered
    match a { Either::A { foo: true } => () }
        //^ error: missing match arm: `B` not covered
    match a {
        Either::A { } => (),
      //^^^^^^^^^ 💡 error: missing structure fields:
      //        | - foo
        Either::B => (),
    }
    match a {
        //^ error: missing match arm: `B` not covered
        Either::A { } => (),
    } //^^^^^^^^^ 💡 error: missing structure fields:
      //        | - foo

    match a {
        Either::A { foo: true } => (),
        Either::A { foo: false } => (),
        Either::B => (),
    }
    match a {
        Either::A { foo: _ } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_fields_out_of_order() {
        check_diagnostics_no_bails(
            r#"
enum Either {
    A { foo: bool, bar: () },
    B,
}

fn main() {
    let a = Either::A { foo: true, bar: () };
    match a {
        //^ error: missing match arm: `B` not covered
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
    }

    match a {
        Either::A { bar: (), foo: false } => (),
        Either::A { foo: true, bar: () } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_record_ellipsis() {
        check_diagnostics_no_bails(
            r#"
enum Either {
    A { foo: bool, bar: bool },
    B,
}

fn main() {
    let a = Either::B;
    match a {
        //^ error: missing match arm: `A { foo: false, .. }` not covered
        Either::A { foo: true, .. } => (),
        Either::B => (),
    }
    match a {
        //^ error: missing match arm: `B` not covered
        Either::A { .. } => (),
    }

    match a {
        Either::A { foo: true, .. } => (),
        Either::A { foo: false, .. } => (),
        Either::B => (),
    }

    match a {
        Either::A { .. } => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn enum_tuple_partial_ellipsis() {
        check_diagnostics_no_bails(
            r#"
enum Either {
    A(bool, bool, bool, bool),
    B,
}

fn main() {
    match Either::B {
        //^^^^^^^^^ error: missing match arm: `A(false, _, _, true)` not covered
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        //^^^^^^^^^ error: missing match arm: `A(false, _, _, false)` not covered
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::B => (),
    }

    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(false, .., true) => (),
        Either::A(false, .., false) => (),
        Either::B => (),
    }
    match Either::B {
        Either::A(true, .., true) => (),
        Either::A(true, .., false) => (),
        Either::A(.., true) => (),
        Either::A(.., false) => (),
        Either::B => (),
    }
}
"#,
        );
    }

    #[test]
    fn never() {
        check_diagnostics_no_bails(
            r#"
enum Never {}

fn enum_(never: Never) {
    match never {}
}
fn enum_ref(never: &Never) {
    match never {}
        //^^^^^ error: missing match arm: type `&Never` is non-empty
}
fn bang(never: !) {
    match never {}
}
"#,
        );
    }

    #[test]
    fn unknown_type() {
        cov_mark::check_count!(validate_match_bailed_out, 1);

        check_diagnostics(
            r#"
enum Option<T> { Some(T), None }

fn main() {
    // `Never` is deliberately not defined so that it's an uninferred type.
    match Option::<Never>::None {
        None => (),
        Some(never) => match never {},
    }
    match Option::<Never>::None {
        //^^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `None` not covered
        Option::Some(_never) => {},
    }
}
"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_end_missing_arm() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(true, _, _)` not covered
        (false, ..) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_at_beginning_missing_arm() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(_, _, true)` not covered
        (.., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn tuple_of_bools_with_ellipsis_in_middle_missing_arm() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(false, _, _)` not covered
        (true, .., false) => (),
    }
}"#,
        );
    }

    #[test]
    fn record_struct() {
        check_diagnostics_no_bails(
            r#"struct Foo { a: bool }
fn main(f: Foo) {
    match f {}
        //^ error: missing match arm: type `Foo` is non-empty
    match f { Foo { a: true } => () }
        //^ error: missing match arm: `Foo { a: false }` not covered
    match &f { Foo { a: true } => () }
        //^^ error: missing match arm: `&Foo { a: false }` not covered
    match f { Foo { a: _ } => () }
    match f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
    match &f {
        Foo { a: true } => (),
        Foo { a: false } => (),
    }
}
"#,
        );
    }

    #[test]
    fn tuple_struct() {
        check_diagnostics_no_bails(
            r#"struct Foo(bool);
fn main(f: Foo) {
    match f {}
        //^ error: missing match arm: type `Foo` is non-empty
    match f { Foo(true) => () }
        //^ error: missing match arm: `Foo(false)` not covered
    match f {
        Foo(true) => (),
        Foo(false) => (),
    }
}
"#,
        );
    }

    #[test]
    fn unit_struct() {
        check_diagnostics_no_bails(
            r#"struct Foo;
fn main(f: Foo) {
    match f {}
        //^ error: missing match arm: type `Foo` is non-empty
    match f { Foo => () }
}
"#,
        );
    }

    #[test]
    fn record_struct_ellipsis() {
        check_diagnostics_no_bails(
            r#"struct Foo { foo: bool, bar: bool }
fn main(f: Foo) {
    match f { Foo { foo: true, .. } => () }
        //^ error: missing match arm: `Foo { foo: false, .. }` not covered
    match f {
        //^ error: missing match arm: `Foo { foo: false, bar: true }` not covered
        Foo { foo: true, .. } => (),
        Foo { bar: false, .. } => ()
    }
    match f { Foo { .. } => () }
    match f {
        Foo { foo: true, .. } => (),
        Foo { foo: false, .. } => ()
    }
}
"#,
        );
    }

    #[test]
    fn internal_or() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    enum Either { A(bool), B }
    match Either::B {
        //^^^^^^^^^ error: missing match arm: `B` not covered
        Either::A(true | false) => (),
    }
}
"#,
        );
    }

    #[test]
    fn no_panic_at_unimplemented_subpattern_type() {
        cov_mark::check_count!(validate_match_bailed_out, 1);

        check_diagnostics(
            r#"
struct S { a: char}
fn main(v: S) {
    match v { S{ a }      => {} }
    match v { S{ a: _x }  => {} }
    match v { S{ a: 'a' } => {} }
    match v { S{..}       => {} }
    match v { _           => {} }
    match v { }
        //^ error: missing match arm: type `S` is non-empty
}
"#,
        );
    }

    #[test]
    fn binding() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match true {
        _x @ true => {}
        false     => {}
    }
    match true { _x @ true => {} }
        //^^^^ error: missing match arm: `false` not covered
}
"#,
        );
    }

    #[test]
    fn binding_ref_has_correct_type() {
        // Asserts `PatKind::Binding(ref _x): bool`, not &bool.
        // If that's not true match checking will panic with "incompatible constructors"
        // FIXME: make facilities to test this directly like `tests::check_infer(..)`
        check_diagnostics_no_bails(
            r#"
enum Foo { A }
fn main() {
    match Foo::A {
        ref _x => {}
        Foo::A => {}
    }
    match (true,) {
        (ref _x,) => {}
        (true,) => {}
    }
}
"#,
        );
    }

    #[test]
    fn enum_non_exhaustive() {
        check_diagnostics_no_bails(
            r#"
//- /lib.rs crate:lib
#[non_exhaustive]
pub enum E { A, B }
fn _local() {
    match E::A { _ => {} }
    match E::A {
        E::A => {}
        E::B => {}
    }
    match E::A {
        E::A | E::B => {}
    }
}

//- /main.rs crate:main deps:lib
use lib::E;
fn main() {
    match E::A { _ => {} }
    match E::A {
        //^^^^ error: missing match arm: `_` not covered
        E::A => {}
        E::B => {}
    }
    match E::A {
        //^^^^ error: missing match arm: `_` not covered
        E::A | E::B => {}
    }
}
"#,
        );
    }

    #[test]
    fn match_guard() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match true {
        true if false => {}
        true          => {}
        false         => {}
    }
    match true {
        //^^^^ error: missing match arm: `true` not covered
        true if false => {}
        false         => {}
    }
}
"#,
        );
    }

    #[test]
    fn pattern_type_is_of_substitution() {
        check_diagnostics_no_bails(
            r#"
struct Foo<T>(T);
struct Bar;
fn main() {
    match Foo(Bar) {
        _ | Foo(Bar) => {}
    }
}
"#,
        );
    }

    #[test]
    fn record_struct_no_such_field() {
        cov_mark::check_count!(validate_match_bailed_out, 1);

        check_diagnostics(
            r#"
struct Foo { }
fn main(f: Foo) {
    match f { Foo { bar } => () }
}
"#,
        );
    }

    #[test]
    fn match_ergonomics_issue_9095() {
        check_diagnostics_no_bails(
            r#"
enum Foo<T> { A(T) }
fn main() {
    match &Foo::A(true) {
        _ => {}
        Foo::A(_) => {}
    }
}
"#,
        );
    }

    #[test]
    fn normalize_field_ty() {
        check_diagnostics_no_bails(
            r"
trait Trait { type Projection; }
enum E {Foo, Bar}
struct A;
impl Trait for A { type Projection = E; }
struct Next<T: Trait>(T::Projection);
static __: () = {
    let n: Next<A> = Next(E::Foo);
    match n { Next(E::Foo) => {} }
    //    ^ error: missing match arm: `Next(Bar)` not covered
    match n { Next(E::Foo | E::Bar) => {} }
    match n { Next(E::Foo | _     ) => {} }
    match n { Next(_      | E::Bar) => {} }
    match n {      _ | Next(E::Bar) => {} }
    match &n { Next(E::Foo | E::Bar) => {} }
    match &n {      _ | Next(E::Bar) => {} }
};",
        );
    }

    #[test]
    fn binding_mode_by_ref() {
        check_diagnostics_no_bails(
            r"
enum E{ A, B }
fn foo() {
    match &E::A {
        E::A => {}
        x => {}
    }
}",
        );
    }

    #[test]
    fn macro_or_pat() {
        check_diagnostics_no_bails(
            r#"
macro_rules! m {
    () => {
        Enum::Type1 | Enum::Type2
    };
}

enum Enum {
    Type1,
    Type2,
    Type3,
}

fn f(ty: Enum) {
    match ty {
        //^^ error: missing match arm: `Type3` not covered
        m!() => (),
    }

    match ty {
        m!() | Enum::Type3 => ()
    }
}
"#,
        );
    }

    #[test]
    fn unexpected_ty_fndef() {
        cov_mark::check!(validate_match_bailed_out);
        check_diagnostics(
            r"
enum Exp {
    Tuple(()),
}
fn f() {
    match __unknown {
        Exp::Tuple => {}
    }
}",
        );
    }

    mod rust_unstable {
        use super::*;

        #[test]
        fn rfc_1872_exhaustive_patterns() {
            check_diagnostics_no_bails(
                r"
//- minicore: option, result
#![feature(exhaustive_patterns)]
enum Void {}
fn test() {
    match None::<!> { None => () }
    match Result::<u8, !>::Ok(2) { Ok(_) => () }
    match Result::<u8, Void>::Ok(2) { Ok(_) => () }
    match (2, loop {}) {}
    match Result::<!, !>::Ok(loop {}) {}
    match (&loop {}) {} // https://github.com/rust-lang/rust/issues/50642#issuecomment-388234919
    //    ^^^^^^^^^^ error: missing match arm: type `&!` is non-empty
}",
            );
        }

        #[test]
        fn rfc_1872_private_uninhabitedness() {
            check_diagnostics_no_bails(
                r"
//- minicore: option
//- /lib.rs crate:lib
#![feature(exhaustive_patterns)]
pub struct PrivatelyUninhabited { private_field: Void }
enum Void {}
fn test_local(x: Option<PrivatelyUninhabited>) {
    match x {}
} //      ^ error: missing match arm: `None` not covered
//- /main.rs crate:main deps:lib
#![feature(exhaustive_patterns)]
fn test(x: Option<lib::PrivatelyUninhabited>) {
    match x {}
    //    ^ error: missing match arm: `None` and `Some(_)` not covered
}",
            );
        }
    }

    mod false_negatives {
        //! The implementation of match checking here is a work in progress. As we roll this out, we
        //! prefer false negatives to false positives (ideally there would be no false positives). This
        //! test module should document known false negatives. Eventually we will have a complete
        //! implementation of match checking and this module will be empty.
        //!
        //! The reasons for documenting known false negatives:
        //!
        //!   1. It acts as a backlog of work that can be done to improve the behavior of the system.
        //!   2. It ensures the code doesn't panic when handling these cases.
        use super::*;

        #[test]
        fn integers() {
            cov_mark::check_count!(validate_match_bailed_out, 1);

            // We don't currently check integer exhaustiveness.
            check_diagnostics(
                r#"
fn main() {
    match 5 {
        10 => (),
        11..20 => (),
    }
}
"#,
            );
        }

        #[test]
        fn reference_patterns_at_top_level() {
            cov_mark::check_count!(validate_match_bailed_out, 1);

            check_diagnostics(
                r#"
//- minicore: copy
fn main() {
    match &false {
        &true => {}
    }
}
            "#,
            );
        }

        #[test]
        fn reference_patterns_in_fields() {
            cov_mark::check_count!(validate_match_bailed_out, 1);
            check_diagnostics(
                r#"
//- minicore: copy
fn main() {
    match (&false,) {
        //^^^^^^^^^ error: missing match arm: `(&false,)` not covered
        (true,) => {}
    }
    match (&false,) {
        (&true,) => {}
    }
}
            "#,
            );
        }
    }
}
