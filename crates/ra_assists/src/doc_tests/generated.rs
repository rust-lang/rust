//! Generated file, do not edit by hand, see `crate/ra_tools/src/codegen`

use super::check;

#[test]
fn doctest_add_derive() {
    check(
        "add_derive",
        r#####"
struct Point {
    x: u32,
    y: u32,<|>
}
"#####,
        r#####"
#[derive()]
struct Point {
    x: u32,
    y: u32,
}
"#####,
    )
}

#[test]
fn doctest_add_explicit_type() {
    check(
        "add_explicit_type",
        r#####"
fn main() {
    let x<|> = 92;
}
"#####,
        r#####"
fn main() {
    let x: i32 = 92;
}
"#####,
    )
}

#[test]
fn doctest_add_impl() {
    check(
        "add_impl",
        r#####"
struct Ctx<T: Clone> {
     data: T,<|>
}
"#####,
        r#####"
struct Ctx<T: Clone> {
     data: T,
}

impl<T: Clone> Ctx<T> {

}
"#####,
    )
}

#[test]
fn doctest_add_impl_default_members() {
    check(
        "add_impl_default_members",
        r#####"
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    Type X = ();
    fn foo(&self) {}<|>

}
"#####,
        r#####"
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    Type X = ();
    fn foo(&self) {}
    fn bar(&self) {}

}
"#####,
    )
}

#[test]
fn doctest_add_impl_missing_members() {
    check(
        "add_impl_missing_members",
        r#####"
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {<|>

}
"#####,
        r#####"
trait T {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl T for () {
    fn foo(&self) { unimplemented!() }

}
"#####,
    )
}

#[test]
fn doctest_apply_demorgan() {
    check(
        "apply_demorgan",
        r#####"
fn main() {
    if x != 4 ||<|> !y {}
}
"#####,
        r#####"
fn main() {
    if !(x == 4 && y) {}
}
"#####,
    )
}

#[test]
fn doctest_change_visibility() {
    check(
        "change_visibility",
        r#####"
fn<|> frobnicate() {}
"#####,
        r#####"
pub(crate) fn frobnicate() {}
"#####,
    )
}

#[test]
fn doctest_convert_to_guarded_return() {
    check(
        "convert_to_guarded_return",
        r#####"
fn main() {
    <|>if cond {
        foo();
        bar();
    }
}
"#####,
        r#####"
fn main() {
    if !cond {
        return;
    }
    foo();
    bar();
}
"#####,
    )
}

#[test]
fn doctest_fill_match_arms() {
    check(
        "fill_match_arms",
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        <|>
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => (),
        Action::Stop => (),
    }
}
"#####,
    )
}

#[test]
fn doctest_flip_binexpr() {
    check(
        "flip_binexpr",
        r#####"
fn main() {
    let _ = 90 +<|> 2;
}
"#####,
        r#####"
fn main() {
    let _ = 2 + 90;
}
"#####,
    )
}

#[test]
fn doctest_flip_comma() {
    check(
        "flip_comma",
        r#####"
fn main() {
    ((1, 2),<|> (3, 4));
}
"#####,
        r#####"
fn main() {
    ((3, 4), (1, 2));
}
"#####,
    )
}

#[test]
fn doctest_inline_local_variable() {
    check(
        "inline_local_variable",
        r#####"
fn main() {
    let x<|> = 1 + 2;
    x * 4;
}
"#####,
        r#####"
fn main() {
    (1 + 2) * 4;
}
"#####,
    )
}
