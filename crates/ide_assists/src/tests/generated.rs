//! Generated file, do not edit by hand, see `xtask/src/codegen`

use super::check_doc_test;

#[test]
fn doctest_add_explicit_type() {
    check_doc_test(
        "add_explicit_type",
        r#####"
fn main() {
    let x$0 = 92;
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
    r#"Hello,$0 World!"#;
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
    type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    type X = ();
    fn foo(&self) {}$0
}
"#####,
        r#####"
trait Trait {
    type X;
    fn foo(&self);
    fn bar(&self) {}
}

impl Trait for () {
    type X = ();
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
    type X;
    fn foo(&self) -> T;
    fn bar(&self) {}
}

impl Trait<u32> for () {$0

}
"#####,
        r#####"
trait Trait<T> {
    type X;
    fn foo(&self) -> T;
    fn bar(&self) {}
}

impl Trait<u32> for () {
    $0type X;

    fn foo(&self) -> u32 {
        todo!()
    }
}
"#####,
    )
}

#[test]
fn doctest_add_lifetime_to_type() {
    check_doc_test(
        "add_lifetime_to_type",
        r#####"
struct Point {
    x: &$0u32,
    y: u32,
}
"#####,
        r#####"
struct Point<'a> {
    x: &'a u32,
    y: u32,
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
    let x = make$0();
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
    if x != 4 ||$0 y < 3.14 {}
}
"#####,
        r#####"
fn main() {
    if !(x == 4 && !(y < 3.14)) {}
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
    let map = HashMap$0::new();
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
fn doctest_change_visibility() {
    check_doc_test(
        "change_visibility",
        r#####"
$0fn frobnicate() {}
"#####,
        r#####"
pub(crate) fn frobnicate() {}
"#####,
    )
}

#[test]
fn doctest_convert_integer_literal() {
    check_doc_test(
        "convert_integer_literal",
        r#####"
const _: i32 = 10$0;
"#####,
        r#####"
const _: i32 = 0b1010;
"#####,
    )
}

#[test]
fn doctest_convert_into_to_from() {
    check_doc_test(
        "convert_into_to_from",
        r#####"
//- minicore: from
impl $0Into<Thing> for usize {
    fn into(self) -> Thing {
        Thing {
            b: self.to_string(),
            a: self
        }
    }
}
"#####,
        r#####"
impl From<usize> for Thing {
    fn from(val: usize) -> Self {
        Thing {
            b: val.to_string(),
            a: val
        }
    }
}
"#####,
    )
}

#[test]
fn doctest_convert_iter_for_each_to_for() {
    check_doc_test(
        "convert_iter_for_each_to_for",
        r#####"
//- minicore: iterators
use core::iter;
fn main() {
    let iter = iter::repeat((9, 2));
    iter.for_each$0(|(x, y)| {
        println!("x: {}, y: {}", x, y);
    });
}
"#####,
        r#####"
use core::iter;
fn main() {
    let iter = iter::repeat((9, 2));
    for (x, y) in iter {
        println!("x: {}, y: {}", x, y);
    }
}
"#####,
    )
}

#[test]
fn doctest_convert_to_guarded_return() {
    check_doc_test(
        "convert_to_guarded_return",
        r#####"
fn main() {
    $0if cond {
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
fn doctest_convert_tuple_struct_to_named_struct() {
    check_doc_test(
        "convert_tuple_struct_to_named_struct",
        r#####"
struct Point$0(f32, f32);

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Point(x, y)
    }

    pub fn x(&self) -> f32 {
        self.0
    }

    pub fn y(&self) -> f32 {
        self.1
    }
}
"#####,
        r#####"
struct Point { field1: f32, field2: f32 }

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Point { field1: x, field2: y }
    }

    pub fn x(&self) -> f32 {
        self.field1
    }

    pub fn y(&self) -> f32 {
        self.field2
    }
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

use foo::*$0;

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
fn doctest_extract_function() {
    check_doc_test(
        "extract_function",
        r#####"
fn main() {
    let n = 1;
    $0let m = n + 2;
    let k = m + n;$0
    let g = 3;
}
"#####,
        r#####"
fn main() {
    let n = 1;
    fun_name(n);
    let g = 3;
}

fn $0fun_name(n: i32) {
    let m = n + 2;
    let k = m + n;
}
"#####,
    )
}

#[test]
fn doctest_extract_struct_from_enum_variant() {
    check_doc_test(
        "extract_struct_from_enum_variant",
        r#####"
enum A { $0One(u32, u32) }
"#####,
        r#####"
struct One(pub u32, pub u32);

enum A { One(One) }
"#####,
    )
}

#[test]
fn doctest_extract_type_alias() {
    check_doc_test(
        "extract_type_alias",
        r#####"
struct S {
    field: $0(u8, u8, u8)$0,
}
"#####,
        r#####"
type $0Type = (u8, u8, u8);

struct S {
    field: Type,
}
"#####,
    )
}

#[test]
fn doctest_extract_variable() {
    check_doc_test(
        "extract_variable",
        r#####"
fn main() {
    $0(1 + 2)$0 * 4;
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
        $0
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    match action {
        $0Action::Move { distance } => todo!(),
        Action::Stop => todo!(),
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
    m::frobnicate$0() {}
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
    let _ = 90 +$0 2;
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
    ((1, 2),$0 (3, 4));
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
fn foo<T: Clone +$0 Copy>() { }
"#####,
        r#####"
fn foo<T: Copy + Clone>() { }
"#####,
    )
}

#[test]
fn doctest_generate_default_from_enum_variant() {
    check_doc_test(
        "generate_default_from_enum_variant",
        r#####"
enum Version {
 Undefined,
 Minor$0,
 Major,
}
"#####,
        r#####"
enum Version {
 Undefined,
 Minor,
 Major,
}

impl Default for Version {
    fn default() -> Self {
        Self::Minor
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_default_from_new() {
    check_doc_test(
        "generate_default_from_new",
        r#####"
struct Example { _inner: () }

impl Example {
    pub fn n$0ew() -> Self {
        Self { _inner: () }
    }
}
"#####,
        r#####"
struct Example { _inner: () }

impl Example {
    pub fn new() -> Self {
        Self { _inner: () }
    }
}

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_deref() {
    check_doc_test(
        "generate_deref",
        r#####"
struct A;
struct B {
   $0a: A
}
"#####,
        r#####"
struct A;
struct B {
   a: A
}

impl std::ops::Deref for B {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.a
    }
}
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
    y: u32,$0
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
fn doctest_generate_enum_as_method() {
    check_doc_test(
        "generate_enum_as_method",
        r#####"
enum Value {
 Number(i32),
 Text(String)$0,
}
"#####,
        r#####"
enum Value {
 Number(i32),
 Text(String),
}

impl Value {
    fn as_text(&self) -> Option<&String> {
        if let Self::Text(v) = self {
            Some(v)
        } else {
            None
        }
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_enum_is_method() {
    check_doc_test(
        "generate_enum_is_method",
        r#####"
enum Version {
 Undefined,
 Minor$0,
 Major,
}
"#####,
        r#####"
enum Version {
 Undefined,
 Minor,
 Major,
}

impl Version {
    /// Returns `true` if the version is [`Minor`].
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_enum_try_into_method() {
    check_doc_test(
        "generate_enum_try_into_method",
        r#####"
enum Value {
 Number(i32),
 Text(String)$0,
}
"#####,
        r#####"
enum Value {
 Number(i32),
 Text(String),
}

impl Value {
    fn try_into_text(self) -> Result<String, Self> {
        if let Self::Text(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_from_impl_for_enum() {
    check_doc_test(
        "generate_from_impl_for_enum",
        r#####"
enum A { $0One(u32) }
"#####,
        r#####"
enum A { One(u32) }

impl From<u32> for A {
    fn from(v: u32) -> Self {
        Self::One(v)
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
    bar$0("", baz());
}

"#####,
        r#####"
struct Baz;
fn baz() -> Baz { Baz }
fn foo() {
    bar("", baz());
}

fn bar(arg: &str, baz: Baz) ${0:-> ()} {
    todo!()
}

"#####,
    )
}

#[test]
fn doctest_generate_getter() {
    check_doc_test(
        "generate_getter",
        r#####"
struct Person {
    nam$0e: String,
}
"#####,
        r#####"
struct Person {
    name: String,
}

impl Person {
    /// Get a reference to the person's name.
    fn $0name(&self) -> &str {
        self.name.as_str()
    }
}
"#####,
    )
}

#[test]
fn doctest_generate_getter_mut() {
    check_doc_test(
        "generate_getter_mut",
        r#####"
struct Person {
    nam$0e: String,
}
"#####,
        r#####"
struct Person {
    name: String,
}

impl Person {
    /// Get a mutable reference to the person's name.
    fn $0name_mut(&mut self) -> &mut String {
        &mut self.name
    }
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
    data: T,$0
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
fn doctest_generate_is_empty_from_len() {
    check_doc_test(
        "generate_is_empty_from_len",
        r#####"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    p$0ub fn len(&self) -> usize {
        self.data.len()
    }
}
"#####,
        r#####"
struct MyStruct { data: Vec<String> }

impl MyStruct {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
     data: T,$0
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
fn doctest_generate_setter() {
    check_doc_test(
        "generate_setter",
        r#####"
struct Person {
    nam$0e: String,
}
"#####,
        r#####"
struct Person {
    name: String,
}

impl Person {
    /// Set the person's name.
    fn set_name(&mut self, name: String) {
        self.name = name;
    }
}
"#####,
    )
}

#[test]
fn doctest_infer_function_return_type() {
    check_doc_test(
        "infer_function_return_type",
        r#####"
fn foo() { 4$02i32 }
"#####,
        r#####"
fn foo() -> i32 { 42i32 }
"#####,
    )
}

#[test]
fn doctest_inline_call() {
    check_doc_test(
        "inline_call",
        r#####"
fn add(a: u32, b: u32) -> u32 { a + b }
fn main() {
    let x = add$0(1, 2);
}
"#####,
        r#####"
fn add(a: u32, b: u32) -> u32 { a + b }
fn main() {
    let x = {
        let a = 1;
        let b = 2;
        a + b
    };
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
    let x$0 = 1 + 2;
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
impl Cursor<'_$0> {
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
    if$0 !y { A } else { B }
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
    "Hello,$0 World!";
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
    r#"Hello,$0 "World!""#;
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
use std::$0fmt::Formatter;
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
        $0Action::Move(..) => foo(),
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
        Action::Move { distance } => $0if distance > 10 { foo() },
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
fn apply<T, U, $0F: FnOnce(T) -> U>(f: F, x: T) -> U {
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
        Action::Move { distance } $0if distance > 10 => foo(),
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
fn doctest_move_module_to_file() {
    check_doc_test(
        "move_module_to_file",
        r#####"
mod $0foo {
    fn t() {}
}
"#####,
        r#####"
mod foo;
"#####,
    )
}

#[test]
fn doctest_pull_assignment_up() {
    check_doc_test(
        "pull_assignment_up",
        r#####"
fn main() {
    let mut foo = 6;

    if true {
        $0foo = 5;
    } else {
        foo = 4;
    }
}
"#####,
        r#####"
fn main() {
    let mut foo = 6;

    foo = if true {
        5
    } else {
        4
    };
}
"#####,
    )
}

#[test]
fn doctest_qualify_path() {
    check_doc_test(
        "qualify_path",
        r#####"
fn main() {
    let map = HashMap$0::new();
}
pub mod std { pub mod collections { pub struct HashMap { } } }
"#####,
        r#####"
fn main() {
    let map = std::collections::HashMap::new();
}
pub mod std { pub mod collections { pub struct HashMap { } } }
"#####,
    )
}

#[test]
fn doctest_remove_dbg() {
    check_doc_test(
        "remove_dbg",
        r#####"
fn main() {
    $0dbg!(92);
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
    r#"Hello,$0 World!"#;
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
    fn feed(&mut$0 self, amount: u32) {}
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
fn frobnicate(x: i32$0) {}

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
const test: Foo = $0Foo {bar: 0, foo: 1}
"#####,
        r#####"
struct Foo {foo: i32, bar: i32};
const test: Foo = Foo {foo: 1, bar: 0}
"#####,
    )
}

#[test]
fn doctest_reorder_impl() {
    check_doc_test(
        "reorder_impl",
        r#####"
trait Foo {
    fn a() {}
    fn b() {}
    fn c() {}
}

struct Bar;
$0impl Foo for Bar {
    fn b() {}
    fn c() {}
    fn a() {}
}
"#####,
        r#####"
trait Foo {
    fn a() {}
    fn b() {}
    fn c() {}
}

struct Bar;
impl Foo for Bar {
    fn a() {}
    fn b() {}
    fn c() {}
}
"#####,
    )
}

#[test]
fn doctest_replace_derive_with_manual_impl() {
    check_doc_test(
        "replace_derive_with_manual_impl",
        r#####"
trait Debug { fn fmt(&self, f: &mut Formatter) -> Result<()>; }
#[derive(Deb$0ug, Display)]
struct S;
"#####,
        r#####"
trait Debug { fn fmt(&self, f: &mut Formatter) -> Result<()>; }
#[derive(Display)]
struct S;

impl Debug for S {
    fn fmt(&self, f: &mut Formatter) -> Result<()> {
        ${0:todo!()}
    }
}
"#####,
    )
}

#[test]
fn doctest_replace_for_loop_with_for_each() {
    check_doc_test(
        "replace_for_loop_with_for_each",
        r#####"
fn main() {
    let x = vec![1, 2, 3];
    for$0 v in x {
        let y = v * 2;
    }
}
"#####,
        r#####"
fn main() {
    let x = vec![1, 2, 3];
    x.into_iter().for_each(|v| {
        let y = v * 2;
    });
}
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
    $0if let Action::Move { distance } = action {
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
fn foo(bar: $0impl Bar) {}
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
    $0let x = compute();
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
fn doctest_replace_match_with_if_let() {
    check_doc_test(
        "replace_match_with_if_let",
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    $0match action {
        Action::Move { distance } => foo(distance),
        _ => bar(),
    }
}
"#####,
        r#####"
enum Action { Move { distance: u32 }, Stop }

fn handle(action: Action) {
    if let Action::Move { distance } = action {
        foo(distance)
    } else {
        bar()
    }
}
"#####,
    )
}

#[test]
fn doctest_replace_qualified_name_with_use() {
    check_doc_test(
        "replace_qualified_name_with_use",
        r#####"
fn process(map: std::collections::$0HashMap<String, String>) {}
"#####,
        r#####"
use std::collections::HashMap;

fn process(map: HashMap<String, String>) {}
"#####,
    )
}

#[test]
fn doctest_replace_string_with_char() {
    check_doc_test(
        "replace_string_with_char",
        r#####"
fn main() {
    find("{$0");
}
"#####,
        r#####"
fn main() {
    find('{');
}
"#####,
    )
}

#[test]
fn doctest_replace_unwrap_with_match() {
    check_doc_test(
        "replace_unwrap_with_match",
        r#####"
//- minicore: result
fn main() {
    let x: Result<i32, i32> = Ok(92);
    let y = x.$0unwrap();
}
"#####,
        r#####"
fn main() {
    let x: Result<i32, i32> = Ok(92);
    let y = match x {
        Ok(it) => it,
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
use std::$0collections::HashMap;
"#####,
        r#####"
use std::{collections::HashMap};
"#####,
    )
}

#[test]
fn doctest_toggle_ignore() {
    check_doc_test(
        "toggle_ignore",
        r#####"
$0#[test]
fn arithmetics {
    assert_eq!(2 + 2, 5);
}
"#####,
        r#####"
#[test]
#[ignore]
fn arithmetics {
    assert_eq!(2 + 2, 5);
}
"#####,
    )
}

#[test]
fn doctest_unmerge_use() {
    check_doc_test(
        "unmerge_use",
        r#####"
use std::fmt::{Debug, Display$0};
"#####,
        r#####"
use std::fmt::{Debug};
use std::fmt::Display;
"#####,
    )
}

#[test]
fn doctest_unwrap_block() {
    check_doc_test(
        "unwrap_block",
        r#####"
fn foo() {
    if true {$0
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

#[test]
fn doctest_wrap_return_type_in_result() {
    check_doc_test(
        "wrap_return_type_in_result",
        r#####"
//- minicore: result
fn foo() -> i32$0 { 42i32 }
"#####,
        r#####"
fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
"#####,
    )
}
