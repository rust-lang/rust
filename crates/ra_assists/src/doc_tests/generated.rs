//! Generated file, do not edit by hand, see `crate/ra_tools/src/codegen`

use super::check;

#[test]
fn doctest_add_custom_impl() {
    check(
        "add_custom_impl",
        r#####"
#[derive(Deb<|>ug, Display)]
struct S;
"#####,
        r#####"
#[derive(Display)]
struct S;

impl Debug for S {

}
"#####,
    )
}

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
fn doctest_add_hash() {
    check(
        "add_hash",
        r#####"
fn main() {
    r#"Hello,<|> World!"#;
}
"#####,
        r#####"
fn main() {
    r##"Hello, World!"##;
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
trait Trait {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    Type X = ();
    fn foo(&self) {}<|>

}
"#####,
        r#####"
trait Trait {
    Type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
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
trait Trait<T> {
    Type X;
    fn foo(&self) -> T;
    fn bar(&self) {}
}

impl Trait<u32> for () {<|>

}
"#####,
        r#####"
trait Trait<T> {
    Type X;
    fn foo(&self) -> T;
    fn bar(&self) {}
}

impl Trait<u32> for () {
    fn foo(&self) -> u32 { unimplemented!() }

}
"#####,
    )
}

#[test]
fn doctest_add_import() {
    check(
        "add_import",
        r#####"
fn process(map: std::collections::<|>HashMap<String, String>) {}
"#####,
        r#####"
use std::collections::HashMap;

fn process(map: HashMap<String, String>) {}
"#####,
    )
}

#[test]
fn doctest_add_new() {
    check(
        "add_new",
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
    fn new(data: T) -> Self { Self { data } }
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
<|>fn frobnicate() {}
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
fn doctest_flip_trait_bound() {
    check(
        "flip_trait_bound",
        r#####"
fn foo<T: Clone +<|> Copy>() { }
"#####,
        r#####"
fn foo<T: Copy + Clone>() { }
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

#[test]
fn doctest_introduce_variable() {
    check(
        "introduce_variable",
        r#####"
fn main() {
    <|>(1 + 2)<|> * 4;
}
"#####,
        r#####"
fn main() {
    let var_name = (1 + 2);
    var_name * 4;
}
"#####,
    )
}

#[test]
fn doctest_invert_if() {
    check(
        "invert_if",
        r#####"
fn main() {
    if<|> !y { A } else { B }
}
"#####,
        r#####"
fn main() {
    if y { B } else { A }
}
"#####,
    )
}

#[test]
fn doctest_make_raw_string() {
    check(
        "make_raw_string",
        r#####"
fn main() {
    "Hello,<|> World!";
}
"#####,
        r#####"
fn main() {
    r#"Hello, World!"#;
}
"#####,
    )
}

#[test]
fn doctest_make_usual_string() {
    check(
        "make_usual_string",
        r#####"
fn main() {
    r#"Hello,<|> "World!""#;
}
"#####,
        r#####"
fn main() {
    "Hello, \"World!\"";
}
"#####,
    )
}

#[test]
fn doctest_merge_match_arms() {
    check(
        "merge_match_arms",
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        <|>Action::Move(..) => foo(),
        Action::Stop => foo(),
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move(..) | Action::Stop => foo(),
    }
}
"#####,
    )
}

#[test]
fn doctest_move_arm_cond_to_match_guard() {
    check(
        "move_arm_cond_to_match_guard",
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => <|>if distance > 10 { foo() },
        _ => (),
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } if distance > 10 => foo(),
        _ => (),
    }
}
"#####,
    )
}

#[test]
fn doctest_move_bounds_to_where_clause() {
    check(
        "move_bounds_to_where_clause",
        r#####"
fn apply<T, U, <|>F: FnOnce(T) -> U>(f: F, x: T) -> U {
    f(x)
}
"#####,
        r#####"
fn apply<T, U, F>(f: F, x: T) -> U where F: FnOnce(T) -> U {
    f(x)
}
"#####,
    )
}

#[test]
fn doctest_move_guard_to_arm_body() {
    check(
        "move_guard_to_arm_body",
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } <|>if distance > 10 => foo(),
        _ => (),
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => if distance > 10 { foo() },
        _ => (),
    }
}
"#####,
    )
}

#[test]
fn doctest_remove_dbg() {
    check(
        "remove_dbg",
        r#####"
fn main() {
    <|>dbg!(92);
}
"#####,
        r#####"
fn main() {
    92;
}
"#####,
    )
}

#[test]
fn doctest_remove_hash() {
    check(
        "remove_hash",
        r#####"
fn main() {
    r#"Hello,<|> World!"#;
}
"#####,
        r#####"
fn main() {
    r"Hello, World!";
}
"#####,
    )
}

#[test]
fn doctest_replace_if_let_with_match() {
    check(
        "replace_if_let_with_match",
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    <|>if let Action::Move { distance } = action {
        foo(distance)
    } else {
        bar()
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        Action::Move { distance } => foo(distance),
        _ => bar(),
    }
}
"#####,
    )
}

#[test]
fn doctest_split_import() {
    check(
        "split_import",
        r#####"
use std::<|>collections::HashMap;
"#####,
        r#####"
use std::{collections::HashMap};
"#####,
    )
}
