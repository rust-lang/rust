use crate::tests::check_no_mismatches;

use super::check;

#[test]
fn function_return_type_mismatch_1() {
    check(
        r#"
fn test() -> &'static str {
    5
  //^ expected &'static str, got i32
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_2() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        return 1;
             //^ expected &'static str, got i32
    }
    "ok"
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_3() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        return "ok";
    }
    1
  //^ expected &'static str, got i32
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_4() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        "ok"
    } else {
        1
      //^ expected &'static str, got i32
    }
}
"#,
    );
}

#[test]
fn function_return_type_mismatch_5() {
    check(
        r#"
fn test(x: bool) -> &'static str {
    if x {
        1
      //^ expected &'static str, got i32
    } else {
        "ok"
    }
}
"#,
    );
}

#[test]
fn non_unit_block_expr_stmt_no_semi() {
    check(
        r#"
fn test(x: bool) {
    if x {
        "notok"
      //^^^^^^^ expected (), got &'static str
    } else {
        "ok"
      //^^^^ expected (), got &'static str
    }
    match x { true => true, false => 0 }
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ expected (), got bool
                                   //^ expected bool, got i32
    ()
}
"#,
    );
}

#[test]
fn no_mismatches_on_atpit() {
    check_no_mismatches(
        r#"
//- minicore: option, sized
#![feature(impl_trait_in_assoc_type)]

trait WrappedAssoc {
    type Assoc;
    fn do_thing(&self) -> Option<Self::Assoc>;
}

struct Foo;
impl WrappedAssoc for Foo {
    type Assoc = impl Sized;

    fn do_thing(&self) -> Option<Self::Assoc> {
        Some(())
    }
}
"#,
    );
    check_no_mismatches(
        r#"
//- minicore: option, sized
#![feature(impl_trait_in_assoc_type)]

trait Trait {
    type Assoc;
    const DEFINE: Option<Self::Assoc>;
}

impl Trait for () {
    type Assoc = impl Sized;
    const DEFINE: Option<Self::Assoc> = Option::Some(());
}
"#,
    );
}

#[test]
fn no_mismatches_with_unresolved_projections() {
    check_no_mismatches(
        r#"
// `Thing` is `{unknown}`
fn create() -> Option<(i32, Thing)> {
    Some((69420, Thing))
}

fn consume() -> Option<()> {
    let (number, thing) = create()?;
    Some(())
}
"#,
    );
}

#[test]
fn method_call_on_field() {
    check(
        r#"
struct S {
    field: fn(f32) -> u32,
    field2: u32
}

fn main() {
    let s = S { field: |_| 0, field2: 0 };
    s.field(0);
         // ^ expected f32, got i32
 // ^^^^^^^^^^ type: u32
    s.field2(0);
          // ^ type: i32
 // ^^^^^^^^^^^ type: {unknown}
    s.not_a_field(0);
               // ^ type: i32
 // ^^^^^^^^^^^^^^^^ type: {unknown}
}
"#,
    );
}

#[test]
fn method_call_on_assoc() {
    check(
        r#"
struct S;

impl S {
    fn not_a_method() -> f32 { 0.0 }
    fn not_a_method2(this: Self, param: f32) -> Self { this }
    fn not_a_method3(param: f32) -> Self { S }
}

fn main() {
    S.not_a_method(0);
 // ^^^^^^^^^^^^^^^^^ type: f32
    S.not_a_method2(0);
                 // ^ expected f32, got i32
 // ^^^^^^^^^^^^^^^^^^ type: S
    S.not_a_method3(0);
 // ^^^^^^^^^^^^^^^^^^ type: S
}
"#,
    );
}
