//! Generated file, do not edit by hand, see `xtask/src/codegen`

use super::check_doc_test;

#[test]
fn doctest_add_custom_impl() {
    check_doc_test(
        "add_custom_impl",
        r#####"
#[derive(Deb<|>ug, Display)]
struct S;
"#####,
        r#####"
#[derive(Display)]
struct S;

impl Debug for S {
    $0
}
"#####,
    )
}

#[test]
fn doctest_add_explicit_type() {
    check_doc_test(
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
    check_doc_test(
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
fn doctest_add_impl_default_members() {
    check_doc_test(
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

    $0fn bar(&self) {}
}
"#####,
    )
}

#[test]
fn doctest_add_impl_missing_members() {
    check_doc_test(
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
    fn foo(&self) -> u32 {
        ${0:todo!()}
    }
}
"#####,
    )
}

#[test]
fn doctest_add_turbo_fish() {
    check_doc_test(
        "add_turbo_fish",
        r#####"
fn make<T>() -> T { todo!() }
fn main() {
    let x = make<|>();
}
"#####,
        r#####"
fn make<T>() -> T { todo!() }
fn main() {
    let x = make::<${0:_}>();
}
"#####,
    )
}

#[test]
fn doctest_apply_demorgan() {
    check_doc_test(
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
fn doctest_auto_import() {
    check_doc_test(
        "auto_import",
        r#####"
fn main() {
    let map = HashMap<|>::new();
}
pub mod std { pub mod collections { pub struct HashMap { } } }
"#####,
        r#####"
use std::collections::HashMap;

fn main() {
    let map = HashMap::new();
}
pub mod std { pub mod collections { pub struct HashMap { } } }
"#####,
    )
}

#[test]
fn doctest_change_return_type_to_result() {
    check_doc_test(
        "change_return_type_to_result",
        r#####"
fn foo() -> i32<|> { 42i32 }
"#####,
        r#####"
fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
"#####,
    )
}

#[test]
fn doctest_change_visibility() {
    check_doc_test(
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
    check_doc_test(
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
fn doctest_expand_glob_import() {
    check_doc_test(
        "expand_glob_import",
        r#####"
mod foo {
    pub struct Bar;
    pub struct Baz;
}

use foo::*<|>;

fn qux(bar: Bar, baz: Baz) {}
"#####,
        r#####"
mod foo {
    pub struct Bar;
    pub struct Baz;
}

use foo::{Baz, Bar};

fn qux(bar: Bar, baz: Baz) {}
"#####,
    )
}

#[test]
fn doctest_extract_struct_from_enum_variant() {
    check_doc_test(
        "extract_struct_from_enum_variant",
        r#####"
enum A { <|>One(u32, u32) }
"#####,
        r#####"
struct One(pub u32, pub u32);

enum A { One(One) }
"#####,
    )
}

#[test]
fn doctest_extract_variable() {
    check_doc_test(
        "extract_variable",
        r#####"
fn main() {
    <|>(1 + 2)<|> * 4;
}
"#####,
        r#####"
fn main() {
    let $0var_name = (1 + 2);
    var_name * 4;
}
"#####,
    )
}

#[test]
fn doctest_fill_match_arms() {
    check_doc_test(
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
        $0Action::Move { distance } => {}
        Action::Stop => {}
    }
}
"#####,
    )
}

#[test]
fn doctest_fix_visibility() {
    check_doc_test(
        "fix_visibility",
        r#####"
mod m {
    fn frobnicate() {}
}
fn main() {
    m::frobnicate<|>() {}
}
"#####,
        r#####"
mod m {
    $0pub(crate) fn frobnicate() {}
}
fn main() {
    m::frobnicate() {}
}
"#####,
    )
}

#[test]
fn doctest_flip_binexpr() {
    check_doc_test(
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
    check_doc_test(
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
    check_doc_test(
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
fn doctest_generate_derive() {
    check_doc_test(
        "generate_derive",
        r#####"
struct Point {
    x: u32,
    y: u32,<|>
}
"#####,
        r#####"
#[derive($0)]
struct Point {
    x: u32,
    y: u32,
}
"#####,
    )
}

#[test]
fn doctest_generate_from_impl_for_enum() {
    check_doc_test(
        "generate_from_impl_for_enum",
        r#####"
enum A { <|>One(u32) }
"#####,
        r#####"
enum A { One(u32) }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        A::One(v)
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_function() {
    check_doc_test(
        "generate_function",
        r#####"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar<|>("", baz());
}

"#####,
        r#####"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar("", baz());
}

fn bar(arg: &str, baz: Baz) {
    ${0:todo!()}
}

"#####,
    )
}

#[test]
fn doctest_generate_impl() {
    check_doc_test(
        "generate_impl",
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
    $0
}
"#####,
    )
}

#[test]
fn doctest_generate_new() {
    check_doc_test(
        "generate_new",
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
    fn $0new(data: T) -> Self { Self { data } }
}

"#####,
    )
}

#[test]
fn doctest_inline_local_variable() {
    check_doc_test(
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
fn doctest_introduce_named_lifetime() {
    check_doc_test(
        "introduce_named_lifetime",
        r#####"
impl Cursor<'_<|>> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
"#####,
        r#####"
impl<'a> Cursor<'a> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
"#####,
    )
}

#[test]
fn doctest_invert_if() {
    check_doc_test(
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
    check_doc_test(
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
    check_doc_test(
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
fn doctest_merge_imports() {
    check_doc_test(
        "merge_imports",
        r#####"
use std::<|>fmt::Formatter;
use std::io;
"#####,
        r#####"
use std::{fmt::Formatter, io};
"#####,
    )
}

#[test]
fn doctest_merge_match_arms() {
    check_doc_test(
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
    check_doc_test(
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
    check_doc_test(
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
    check_doc_test(
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
        Action::Move { distance } => if distance > 10 {
            foo()
        },
        _ => (),
    }
}
"#####,
    )
}

#[test]
fn doctest_remove_dbg() {
    check_doc_test(
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
    check_doc_test(
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
fn doctest_remove_mut() {
    check_doc_test(
        "remove_mut",
        r#####"
impl Walrus {
    fn feed(&mut<|> self, amount: u32) {}
}
"#####,
        r#####"
impl Walrus {
    fn feed(&self, amount: u32) {}
}
"#####,
    )
}

#[test]
fn doctest_remove_unused_param() {
    check_doc_test(
        "remove_unused_param",
        r#####"
fn frobnicate(x: i32<|>) {}

fn main() {
    frobnicate(92);
}
"#####,
        r#####"
fn frobnicate() {}

fn main() {
    frobnicate();
}
"#####,
    )
}

#[test]
fn doctest_reorder_fields() {
    check_doc_test(
        "reorder_fields",
        r#####"
struct Foo {foo: i32, bar: i32};
const test: Foo = <|>Foo {bar: 0, foo: 1}
"#####,
        r#####"
struct Foo {foo: i32, bar: i32};
const test: Foo = Foo {foo: 1, bar: 0}
"#####,
    )
}

#[test]
fn doctest_replace_if_let_with_match() {
    check_doc_test(
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
fn doctest_replace_impl_trait_with_generic() {
    check_doc_test(
        "replace_impl_trait_with_generic",
        r#####"
fn foo(bar: <|>impl Bar) {}
"#####,
        r#####"
fn foo<B: Bar>(bar: B) {}
"#####,
    )
}

#[test]
fn doctest_replace_let_with_if_let() {
    check_doc_test(
        "replace_let_with_if_let",
        r#####"
enum Option<T> { Some(T), None }

fn main(action: Action) {
    <|>let x = compute();
}

fn compute() -> Option<i32> { None }
"#####,
        r#####"
enum Option<T> { Some(T), None }

fn main(action: Action) {
    if let Some(x) = compute() {
    }
}

fn compute() -> Option<i32> { None }
"#####,
    )
}

#[test]
fn doctest_replace_qualified_name_with_use() {
    check_doc_test(
        "replace_qualified_name_with_use",
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
fn doctest_replace_unwrap_with_match() {
    check_doc_test(
        "replace_unwrap_with_match",
        r#####"
enum Result<T, E> { Ok(T), Err(E) }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = x.<|>unwrap();
}
"#####,
        r#####"
enum Result<T, E> { Ok(T), Err(E) }
fn main() {
    let x: Result<i32, i32> = Result::Ok(92);
    let y = match x {
        Ok(a) => a,
        $0_ => unreachable!(),
    };
}
"#####,
    )
}

#[test]
fn doctest_split_import() {
    check_doc_test(
        "split_import",
        r#####"
use std::<|>collections::HashMap;
"#####,
        r#####"
use std::{collections::HashMap};
"#####,
    )
}

#[test]
fn doctest_unwrap_block() {
    check_doc_test(
        "unwrap_block",
        r#####"
fn foo() {
    if true {<|>
        println!("foo");
    }
}
"#####,
        r#####"
fn foo() {
    println!("foo");
}
"#####,
    )
}
