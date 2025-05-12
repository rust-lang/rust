use std::time::Instant;

use expect_test::{ExpectFile, expect_file};
use ide_db::SymbolKind;
use span::Edition;
use test_utils::{AssertLinear, bench, bench_fixture, skip_slow_tests};

use crate::{FileRange, HighlightConfig, HlTag, TextRange, fixture};

const HL_CONFIG: HighlightConfig = HighlightConfig {
    strings: true,
    punctuation: true,
    specialize_punctuation: true,
    specialize_operator: true,
    operator: true,
    inject_doc_comment: true,
    macro_bang: true,
    syntactic_name_ref_highlighting: false,
};

#[test]
fn attributes() {
    check_highlighting(
        r#"
//- proc_macros: identity
//- minicore: derive, copy, default
#[allow(dead_code)]
#[rustfmt::skip]
#[proc_macros::identity]
#[derive(Default)]
/// This is a doc comment
// This is a normal comment
/// This is a doc comment
#[derive(Copy)]
// This is another normal comment
/// This is another doc comment
// This is another normal comment
#[derive(Copy, Unresolved)]
// The reason for these being here is to test AttrIds
enum Foo {
    #[default]
    Bar
}
"#,
        expect_file!["./test_data/highlight_attributes.html"],
        false,
    );
}

#[test]
fn macros() {
    check_highlighting(
        r#"
//- proc_macros: mirror, identity, derive_identity
//- minicore: fmt, include, concat
//- /lib.rs crate:lib
use proc_macros::{mirror, identity, DeriveIdentity};

mirror! {
    {
        ,i32 :x pub
        ,i32 :y pub
    } Foo struct
}
macro_rules! def_fn {
    ($($tt:tt)*) => {$($tt)*}
}

def_fn! {
    fn bar() -> u32 {
        100
    }
}

macro_rules! dont_color_me_braces {
    () => {0}
}

macro_rules! noop {
    ($expr:expr) => {
        $expr
    }
}

/// textually shadow previous definition
macro_rules! noop {
    ($expr:expr) => {
        $expr
    }
}

macro_rules! keyword_frag {
    ($type:ty) => ($type)
}

macro with_args($i:ident) {
    $i
}

macro without_args {
    ($i:ident) => {
        $i
    }
}

macro_rules! id {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

include!(concat!("foo/", "foo.rs"));

struct S<T>(T);
fn main() {
    struct TestLocal;
    // regression test, TestLocal here used to not resolve
    let _: S<id![TestLocal]>;

    format_args!("Hello, {}!", (92,).0);
    dont_color_me_braces!();
    noop!(noop!(1));
}
//- /foo/foo.rs crate:foo
mod foo {}
use self::foo as bar;
"#,
        expect_file!["./test_data/highlight_macros.html"],
        false,
    );
}

/// If what you want to test feels like a specific entity consider making a new test instead,
/// this test fixture here in fact should shrink instead of grow ideally.
#[test]
fn test_highlighting() {
    check_highlighting(
        r#"
//- minicore: derive, copy, fn
//- /main.rs crate:main deps:foo
use inner::{self as inner_mod};
mod inner {}

struct Foo {
    x: u32,
}

trait Bar {
    fn bar(&self) -> i32;
}

impl Bar for Foo {
    fn bar(&self) -> i32 {
        self.x
    }
}

impl Foo {
    fn baz(mut self, f: Foo) -> i32 {
        f.baz(self)
    }

    fn qux(&mut self) {
        self.x = 0;
    }

    fn quop(&self) -> i32 {
        self.x
    }
}

use self::FooCopy::{self as BarCopy};

#[derive(Copy)]
struct FooCopy {
    x: u32,
}

impl FooCopy {
    fn baz(self, f: FooCopy) -> u32 {
        f.baz(self)
    }

    fn qux(&mut self) {
        self.x = 0;
    }

    fn quop(&self) -> u32 {
        self.x
    }
}

fn str() {
    str();
}

fn foo<'a, T>() -> T {
    foo::<'a, i32>()
}

fn never() -> ! {
    loop {}
}

fn const_param<const FOO: usize>() -> usize {
    const_param::<{ FOO }>();
    FOO
}

use core::ops::Fn;
fn baz<F: Fn() -> ()>(f: F) {
    f()
}

fn foobar() -> impl Copy {}

fn foo() {
    let bar = foobar();
}

// comment
fn main() {
    let mut x = 42;
    x += 1;
    let y = &mut x;
    let z = &y;

    let Foo { x: z, y } = Foo { x: z, y };

    y;

    let mut foo = Foo { x, y: x };
    let foo2 = Foo { x, y: x };
    foo.quop();
    foo.qux();
    foo.baz(foo2);

    let mut copy = FooCopy { x };
    copy.quop();
    copy.qux();
    copy.baz(copy);

    let a = |x| x;
    let bar = Foo::baz;

    let baz = (-42,);
    let baz = -baz.0;

    let _ = !true;

    'foo: loop {
        break 'foo;
        continue 'foo;
    }
}

enum Option<T> {
    Some(T),
    None,
}
use Option::*;

impl<T> Option<T> {
    fn and<U>(self, other: Option<U>) -> Option<(T, U)> {
        match other {
            None => unimplemented!(),
            Nope => Nope,
        }
    }
}

async fn learn_and_sing() {
    let song = learn_song().await;
    sing_song(song).await;
}

async fn async_main() {
    let f1 = learn_and_sing();
    let f2 = dance();
    futures::join!(f1, f2);
}

fn use_foo_items() {
    let bob = foo::Person {
        name: "Bob",
        age: foo::consts::NUMBER,
    };

    let control_flow = foo::identity(foo::ControlFlow::Continue);

    if control_flow.should_die() {
        foo::die!();
    }
}

pub enum Bool { True, False }

impl Bool {
    pub const fn to_primitive(self) -> bool {
        true
    }
}
const USAGE_OF_BOOL: bool = Bool::True.to_primitive();

trait Baz {
    type Qux;
}

fn baz<T>(t: T)
where
    T: Baz,
    <T as Baz>::Qux: Bar {}

fn gp_shadows_trait<Baz: Bar>() {
    Baz::bar;
}

//- /foo.rs crate:foo
pub struct Person {
    pub name: &'static str,
    pub age: u8,
}

pub enum ControlFlow {
    Continue,
    Die,
}

impl ControlFlow {
    pub fn should_die(self) -> bool {
        matches!(self, ControlFlow::Die)
    }
}

pub fn identity<T>(x: T) -> T { x }

pub mod consts {
    pub const NUMBER: i64 = 92;
}

macro_rules! die {
    () => {
        panic!();
    };
}
"#,
        expect_file!["./test_data/highlight_general.html"],
        false,
    );
}

#[test]
fn test_lifetime_highlighting() {
    check_highlighting(
        r#"
//- minicore: derive

#[derive()]
struct Foo<'a, 'b, 'c> where 'a: 'a, 'static: 'static {
    field: &'a (),
    field2: &'static (),
}
impl<'a> Foo<'_, 'a, 'static>
where
    'a: 'a,
    'static: 'static
{}
"#,
        expect_file!["./test_data/highlight_lifetimes.html"],
        false,
    );
}

#[test]
fn test_keyword_highlighting() {
    for edition in Edition::iter() {
        check_highlighting(
            &(format!("//- /main.rs crate:main edition:{edition}")
                + r#"
extern crate self;

use crate;
use self;
mod __ {
    use super::*;
}

macro_rules! void {
    ($($tt:tt)*) => {discard!($($tt:tt)*)}
}

struct __ where Self:;
fn __(_: Self) {}
void!(Self);

// edition dependent
void!(try async await gen);
// edition and context dependent
void!(dyn);
// builtin custom syntax
void!(builtin offset_of format_args asm);
// contextual
void!(macro_rules, union, default, raw, auto, yeet);
// reserved
void!(abstract become box do final macro override priv typeof unsized virtual yield);
void!('static 'self 'unsafe)
"#),
            expect_file![format!("./test_data/highlight_keywords_{edition}.html")],
            false,
        );
    }
}

#[test]
fn test_keyword_macro_edition_highlighting() {
    check_highlighting(
        r#"
//- /main.rs crate:main edition:2018 deps:lib2015,lib2024
lib2015::void_2015!(try async await gen);
lib2024::void_2024!(try async await gen);
//- /lib2015.rs crate:lib2015 edition:2015
#[macro_export]
macro_rules! void_2015 {
    ($($tt:tt)*) => {discard!($($tt:tt)*)}
}

//- /lib2024.rs crate:lib2024 edition:2024
#[macro_export]
macro_rules! void_2024 {
    ($($tt:tt)*) => {discard!($($tt:tt)*)}
}

"#,
        expect_file![format!("./test_data/highlight_keywords_macros.html")],
        false,
    );
}

#[test]
fn test_string_highlighting() {
    // The format string detection is based on macro-expansion,
    // thus, we have to copy the macro definition from `std`
    check_highlighting(
        r#"
//- minicore: fmt, assert, asm, concat, panic
macro_rules! println {
    ($($arg:tt)*) => ({
        $crate::io::_print(format_args_nl!($($arg)*));
    })
}

mod panic {
    pub macro panic_2015 {
        () => (
            panic("explicit panic")
        ),
        ($msg:literal $(,)?) => (
            panic($msg)
        ),
        // Use `panic_str` instead of `panic_display::<&str>` for non_fmt_panic lint.
        ($msg:expr $(,)?) => (
            panic_str($msg)
        ),
        // Special-case the single-argument case for const_panic.
        ("{}", $arg:expr $(,)?) => (
            panic_display(&$arg)
        ),
        ($fmt:expr, $($arg:tt)+) => (
            panic_fmt(const_format_args!($fmt, $($arg)+))
        ),
    }
}

macro_rules! toho {
    () => ($crate::panic!("not yet implemented"));
    ($($arg:tt)+) => ($crate::panic!("not yet implemented: {}", format_args!($($arg)+)));
}

macro_rules! reuse_twice {
    ($literal:literal) => {{stringify!($literal); format_args!($literal)}};
}

use foo::bar as baz;
trait Bar = Baz;
trait Foo = Bar;

fn main() {
    let a = '\n';
    let a = '\t';
    let a = '\e'; // invalid escape
    let a = 'e';
    let a = ' ';
    let a = '\u{48}';
    let a = '\u{4823}';
    let a = '\x65';
    let a = '\x00';

    let a = b'\xFF';

    println!("Hello {{Hello}}");
    // from https://doc.rust-lang.org/std/fmt/index.html
    println!("Hello");                 // => "Hello"
    println!("Hello, {}!", "world");   // => "Hello, world!"
    println!("The number is {}", 1);   // => "The number is 1"
    println!("{:?}", (3, 4));          // => "(3, 4)"
    println!("{value}", value=4);      // => "4"
    println!("{} {}", 1, 2);           // => "1 2"
    println!("{:04}", 42);             // => "0042" with leading zerosV
    println!("{1} {} {0} {}", 1, 2);   // => "2 1 1 2"
    println!("{argument}", argument = "test");   // => "test"
    println!("{name} {}", 1, name = 2);          // => "2 1"
    println!("{a} {c} {b}", a="a", b='b', c=3);  // => "a 3 b"
    println!("{{{}}}", 2);                       // => "{2}"
    println!("Hello {:5}!", "x");
    println!("Hello {:1$}!", "x", 5);
    println!("Hello {1:0$}!", 5, "x");
    println!("Hello {:width$}!", "x", width = 5);
    println!("Hello {:<5}!", "x");
    println!("Hello {:-<5}!", "x");
    println!("Hello {:^5}!", "x");
    println!("Hello {:>5}!", "x");
    println!("Hello {:+}!", 5);
    println!("{:#x}!", 27);
    println!("Hello {:05}!", 5);
    println!("Hello {:05}!", -5);
    println!("{:#010x}!", 27);
    println!("Hello {0} is {1:.5}", "x", 0.01);
    println!("Hello {1} is {2:.0$}", 5, "x", 0.01);
    println!("Hello {0} is {2:.1$}", "x", 5, 0.01);
    println!("Hello {} is {:.*}",    "x", 5, 0.01);
    println!("Hello {} is {2:.*}",   "x", 5, 0.01);
    println!("Hello {} is {number:.prec$}", "x", prec = 5, number = 0.01);
    println!("{}, `{name:.*}` has 3 fractional digits", "Hello", 3, name=1234.56);
    println!("{}, `{name:.*}` has 3 characters", "Hello", 3, name="1234.56");
    println!("{}, `{name:>8.*}` has 3 right-aligned characters", "Hello", 3, name="1234.56");

    let _ = "{}"
    let _ = "{{}}";

    println!("Hello {{}}");
    println!("{{ Hello");
    println!("Hello }}");
    println!("{{Hello}}");
    println!("{{ Hello }}");
    println!("{{Hello }}");
    println!("{{ Hello}}");

    println!(r"Hello, {}!", "world");

    // escape sequences
    println!("Hello\nWorld");
    println!("\u{48}\x65\x6C\x6C\x6F World");

    let _ = "\x28\x28\x00\x63\xFF\u{FF}\n"; // invalid non-UTF8 escape sequences
    let _ = b"\x28\x28\x00\x63\xFF\u{FF}\n"; // valid bytes, invalid unicodes
    let _ = c"\u{FF}\xFF"; // valid bytes, valid unicodes
    let backslash = r"\\";

    println!("{\x41}", A = 92);
    println!("{ничоси}", ничоси = 92);

    println!("{:x?} {} ", thingy, n2);
    panic!("{}", 0);
    panic!("more {}", 1);
    assert!(true, "{}", 1);
    assert!(true, "{} asdasd", 1);
    toho!("{}fmt", 0);
    let i: u64 = 3;
    let o: u64;
    core::arch::asm!(
        "mov {0}, {1}",
        "add {0}, 5",
        out(reg) o,
        in(reg) i,
    );

    const CONSTANT: () = ():
    let mut m = ();
    format_args!(concat!("{}"), "{}");
    format_args!("{} {} {} {} {} {} {backslash} {CONSTANT} {m}", backslash, format_args!("{}", 0), foo, "bar", toho!(), backslash);
    reuse_twice!("{backslash}");
}"#,
        expect_file!["./test_data/highlight_strings.html"],
        false,
    );
}

#[test]
fn test_unsafe_highlighting() {
    check_highlighting(
        r#"
//- minicore: sized, asm
macro_rules! id {
    ($($tt:tt)*) => {
        $($tt)*
    };
}
macro_rules! unsafe_deref {
    () => {
        *(&() as *const ())
    };
}

union Union {
    field: u32,
}

struct Struct { field: i32 }

static mut MUT_GLOBAL: Struct = Struct { field: 0 };
unsafe fn unsafe_fn() {}

impl Struct {
    unsafe fn unsafe_method(&self) {}
}

unsafe trait UnsafeTrait {}
unsafe impl UnsafeTrait for Union {}
impl !UnsafeTrait for () {}

fn unsafe_trait_bound<T: UnsafeTrait>(_: T) {}

extern {
    static EXTERN_STATIC: ();
}

fn main() {
    let x: *const usize;
    let u: Union;

    // id should be safe here, but unsafe_deref should not
    id! {
        unsafe { unsafe_deref!() }
    };

    unsafe {
        // unsafe macro calls
        unsafe_deref!();
        id! { unsafe_deref!() };

        // unsafe fn and method calls
        unsafe_fn();
        self::unsafe_fn();
        (unsafe_fn as unsafe fn())();
        Struct { field: 0 }.unsafe_method();

        u.field;
        &u.field;
        &raw const u.field;
        // this should be safe!
        let Union { field: _ };
        // but not these
        let Union { field };
        let Union { field: field };
        let Union { field: ref field };
        let Union { field: (_ | ref field) };

        // unsafe deref
        *&raw const*&*x;

        // unsafe access to a static mut
        MUT_GLOBAL.field;
        &MUT_GLOBAL.field;
        &raw const MUT_GLOBAL.field;
        MUT_GLOBAL;
        &MUT_GLOBAL;
        &raw const MUT_GLOBAL;
        EXTERN_STATIC;
        &EXTERN_STATIC;
        &raw const EXTERN_STATIC;

        core::arch::asm!(
            "push {base}",
            base = const 0
        );
    }
}
"#,
        expect_file!["./test_data/highlight_unsafe.html"],
        false,
    );
}

#[test]
fn test_const_highlighting() {
    check_highlighting(
        r#"
macro_rules! id {
    ($($tt:tt)*) => {
        $($tt)*
    };
}
const CONST_ITEM: *const () = &raw const ();
const fn const_fn<const CONST_PARAM: ()>(const {}: const fn()) where (): const ConstTrait {
    CONST_ITEM;
    CONST_PARAM;
    const {
        const || {}
    }
    id!(
        CONST_ITEM;
        CONST_PARAM;
        const {
            const || {}
        };
        &raw const ();
        const
    );
    ().assoc_const_method();
}
trait ConstTrait {
    const ASSOC_CONST: () = ();
    const fn assoc_const_fn() {}
    const fn assoc_const_method(self) {}
}
impl const ConstTrait for () {
    const ASSOC_CONST: () = ();
    const fn assoc_const_fn() {}
}

macro_rules! unsafe_deref {
    () => {
        *(&() as *const ())
    };
}
"#,
        expect_file!["./test_data/highlight_const.html"],
        false,
    );
}

#[test]
fn test_highlight_doc_comment() {
    check_highlighting(
        r#"
//- /main.rs
//! This is a module to test doc injection.
//! ```
//! fn test() {}
//! ```

//! Syntactic name ref highlighting testing
//! ```rust
//! extern crate self;
//! extern crate other as otter;
//! extern crate core;
//! trait T { type Assoc; }
//! fn f<Arg>() -> use<Arg> where (): T<Assoc = ()> {}
//! ```
mod outline_module;

/// ```
/// let _ = "early doctests should not go boom";
/// ```
struct Foo {
    bar: bool,
}

/// This is an impl of [`Foo`] with a code block.
///
/// ```
/// fn foo() {
///
/// }
/// ```
impl Foo {
    /// ```
    /// let _ = "Call me
    //    KILLER WHALE
    ///     Ishmael.";
    /// ```
    pub const bar: bool = true;

    /// Constructs a new `Foo`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(unused_mut)]
    /// let mut foo: Foo = Foo::new();
    /// ```
    pub const fn new() -> Foo {
        Foo { bar: true }
    }

    /// `bar` method on `Foo`.
    ///
    /// # Examples
    ///
    /// ```
    /// use x::y;
    ///
    /// let foo = Foo::new();
    ///
    /// // calls bar on foo
    /// assert!(foo.bar());
    ///
    /// let bar = foo.bar || Foo::bar;
    ///
    /// /* multi-line
    ///        comment */
    ///
    /// let multi_line_string = "Foo
    ///   bar\n
    ///          ";
    ///
    /// ```
    ///
    /// ```rust,no_run
    /// let foobar = Foo::new().bar();
    /// ```
    ///
    /// ~~~rust,no_run
    /// // code block with tilde.
    /// let foobar = Foo::new().bar();
    /// ~~~
    ///
    /// ```
    /// // functions
    /// fn foo<T, const X: usize>(arg: i32) {
    ///     let x: T = X;
    /// }
    /// ```
    ///
    /// ```sh
    /// echo 1
    /// ```
    pub fn foo(&self) -> bool {
        true
    }
}

/// [`Foo`](Foo) is a struct
/// This function is > [`all_the_links`](all_the_links) <
/// [`noop`](noop) is a macro below
/// [`Item`] is a struct in the module [`module`]
///
/// [`Item`]: module::Item
/// [mix_and_match]: ThisShouldntResolve
pub fn all_the_links() {}

pub mod module {
    pub struct Item;
}

/// ```
/// macro_rules! noop { ($expr:expr) => { $expr }}
/// noop!(1);
/// ```
macro_rules! noop {
    ($expr:expr) => {
        $expr
    }
}

/// ```rust
/// let _ = example(&[1, 2, 3]);
/// ```
///
/// ```
/// loop {}
#[cfg_attr(not(feature = "false"), doc = "loop {}")]
#[doc = "loop {}"]
/// ```
///
#[cfg_attr(feature = "alloc", doc = "```rust")]
#[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
/// let _ = example(&alloc::vec![1, 2, 3]);
/// ```
pub fn mix_and_match() {}

/**
It is beyond me why you'd use these when you got ///
```rust
let _ = example(&[1, 2, 3]);
```
[`block_comments2`] tests these with indentation
 */
pub fn block_comments() {}

/**
    Really, I don't get it
    ```rust
    let _ = example(&[1, 2, 3]);
    ```
    [`block_comments`] tests these without indentation
*/
pub fn block_comments2() {}

//- /outline_module.rs
//! This is an outline module whose purpose is to test that its inline attribute injection does not
//! spill into its parent.
//! ```
//! fn test() {}
//! ```
"#,
        expect_file!["./test_data/highlight_doctest.html"],
        false,
    );
}

#[test]
fn test_extern_crate() {
    check_highlighting(
        r#"
//- /main.rs crate:main deps:std,alloc,test,proc_macro extern-prelude:std,alloc
extern crate self as this;
extern crate std;
extern crate alloc as abc;
extern crate unresolved as definitely_unresolved;
extern crate unresolved as _;
extern crate test as opt_in_crate;
extern crate test as _;
extern crate proc_macro;
//- /std/lib.rs crate:std
pub struct S;
//- /alloc/lib.rs crate:alloc
pub struct A;
//- /test/lib.rs crate:test
pub struct T;
//- /proc_macro/lib.rs crate:proc_macro
pub struct ProcMacro;
"#,
        expect_file!["./test_data/highlight_extern_crate.html"],
        false,
    );
}

#[test]
fn test_crate_root() {
    check_highlighting(
        r#"
//- minicore: iterators
//- /main.rs crate:main deps:foo
extern crate foo;
use core::iter;

pub const NINETY_TWO: u8 = 92;

use foo as foooo;

pub(crate) fn main() {
    let baz = iter::repeat(92);
}

mod bar {
    pub(in super) const FORTY_TWO: u8 = 42;

    mod baz {
        use super::super::NINETY_TWO;
        use crate::foooo::Point;

        pub(in super::super) const TWENTY_NINE: u8 = 29;
    }
}
//- /foo.rs crate:foo
struct Point {
    x: u8,
    y: u8,
}

mod inner {
    pub(super) fn swap(p: crate::Point) -> crate::Point {
        crate::Point { x: p.y, y: p.x }
    }
}
"#,
        expect_file!["./test_data/highlight_crate_root.html"],
        false,
    );
}

#[test]
fn test_default_library() {
    check_highlighting(
        r#"
//- minicore: option, iterators
use core::iter;

fn main() {
    let foo = Some(92);
    let nums = iter::repeat(foo.unwrap());
}
"#,
        expect_file!["./test_data/highlight_default_library.html"],
        false,
    );
}

#[test]
fn test_associated_function() {
    check_highlighting(
        r#"
fn not_static() {}

struct foo {}

impl foo {
    pub fn is_static() {}
    pub fn is_not_static(&self) {}
}

trait t {
    fn t_is_static() {}
    fn t_is_not_static(&self) {}
}

impl t for foo {
    pub fn is_static() {}
    pub fn is_not_static(&self) {}
}
"#,
        expect_file!["./test_data/highlight_assoc_functions.html"],
        false,
    )
}

#[test]
fn test_injection() {
    check_highlighting(
        r##"
fn fixture(#[rust_analyzer::rust_fixture] ra_fixture: &str) {}

fn main() {
    fixture(r#"
trait Foo {
    fn foo() {
        println!("2 + 2 = {}", 4);
    }
}"#
    );
    fixture(r"
fn foo() {
    foo(\$0{
        92
    }\$0)
}"
    );
}
"##,
        expect_file!["./test_data/highlight_injection.html"],
        false,
    );
}

#[test]
fn test_operators() {
    check_highlighting(
        r##"
fn main() {
    1 + 1 - 1 * 1 / 1 % 1 | 1 & 1 ! 1 ^ 1 >> 1 << 1;
    let mut a = 0;
    a += 1;
    a -= 1;
    a *= 1;
    a /= 1;
    a %= 1;
    a |= 1;
    a &= 1;
    a ^= 1;
    a >>= 1;
    a <<= 1;
}
"##,
        expect_file!["./test_data/highlight_operators.html"],
        false,
    );
}

#[test]
fn test_mod_hl_injection() {
    check_highlighting(
        r##"
//- /foo.rs
//! [Struct]
//! This is an intra doc injection test for modules
//! [Struct]
//! This is an intra doc injection test for modules

pub struct Struct;
//- /lib.rs crate:foo
/// [crate::foo::Struct]
/// This is an intra doc injection test for modules
/// [crate::foo::Struct]
/// This is an intra doc injection test for modules
mod foo;
"##,
        expect_file!["./test_data/highlight_module_docs_inline.html"],
        false,
    );
    check_highlighting(
        r##"
//- /lib.rs crate:foo
/// [crate::foo::Struct]
/// This is an intra doc injection test for modules
/// [crate::foo::Struct]
/// This is an intra doc injection test for modules
mod foo;
//- /foo.rs
//! [Struct]
//! This is an intra doc injection test for modules
//! [Struct]
//! This is an intra doc injection test for modules

pub struct Struct;
"##,
        expect_file!["./test_data/highlight_module_docs_outline.html"],
        false,
    );
}

// Rainbow highlighting uses a deterministic hash (fxhash) but the hashing does differ
// depending on the pointer width so only runs this on 64-bit targets.
#[cfg(target_pointer_width = "64")]
#[test]
fn test_rainbow_highlighting() {
    check_highlighting(
        r#"
fn main() {
    let hello = "hello";
    let x = hello.to_string();
    let y = hello.to_string();

    let x = "other color please!";
    let y = x.to_string();
}

fn bar() {
    let mut hello = "hello";
}
"#,
        expect_file!["./test_data/highlight_rainbow.html"],
        true,
    );
}

#[test]
fn test_block_mod_items() {
    check_highlighting(
        r#"
macro_rules! foo {
    ($foo:ident) => {
        mod y {
            pub struct $foo;
        }
    };
}
fn main() {
    foo!(Foo);
    mod module {
        foo!(Bar);
        fn func(_: y::Bar) {
            mod inner {
                struct Innerest<const C: usize> { field: [(); {C}] }
            }
        }
    }
}
"#,
        expect_file!["./test_data/highlight_block_mod_items.html"],
        false,
    );
}

#[test]
fn test_ranges() {
    let (analysis, file_id) = fixture::file(
        r#"
#[derive(Clone, Debug)]
struct Foo {
    pub x: i32,
    pub y: i32,
}
"#,
    );

    // The "x"
    let highlights = &analysis
        .highlight_range(
            HL_CONFIG,
            FileRange { file_id, range: TextRange::at(45.into(), 1.into()) },
        )
        .unwrap();

    assert_eq!(&highlights[0].highlight.to_string(), "field.declaration.public");
}

#[test]
fn ranges_sorted() {
    let (analysis, file_id) = fixture::file(
        r#"
#[foo(bar = "bar")]
macro_rules! test {}
}"#
        .trim(),
    );
    let _ = analysis.highlight(HL_CONFIG, file_id).unwrap();
}

#[test]
fn highlight_callable_no_crash() {
    // regression test for #13838.
    let (analysis, file_id) = fixture::file(
        r#"
//- minicore: fn, sized
impl<A, F: ?Sized> FnOnce<A> for &F
where
    F: Fn<A>,
{
    type Output = F::Output;
}

trait Trait {}
fn foo(x: &fn(&dyn Trait)) {}
"#,
    );
    let _ = analysis.highlight(HL_CONFIG, file_id).unwrap();
}

/// Highlights the code given by the `ra_fixture` argument, renders the
/// result as HTML, and compares it with the HTML file given as `snapshot`.
/// Note that the `snapshot` file is overwritten by the rendered HTML.
fn check_highlighting(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: ExpectFile,
    rainbow: bool,
) {
    let (analysis, file_id) = fixture::file(ra_fixture.trim());
    let actual_html = &analysis.highlight_as_html(file_id, rainbow).unwrap();
    expect.assert_eq(actual_html)
}

#[test]
fn benchmark_syntax_highlighting_long_struct() {
    if skip_slow_tests() {
        return;
    }

    let fixture = bench_fixture::big_struct();
    let (analysis, file_id) = fixture::file(&fixture);

    let hash = {
        let _pt = bench("syntax highlighting long struct");
        analysis
            .highlight(HL_CONFIG, file_id)
            .unwrap()
            .iter()
            .filter(|it| it.highlight.tag == HlTag::Symbol(SymbolKind::Struct))
            .count()
    };
    assert_eq!(hash, 2001);
}

#[test]
fn syntax_highlighting_not_quadratic() {
    if skip_slow_tests() {
        return;
    }

    let mut al = AssertLinear::default();
    while al.next_round() {
        for i in 6..=10 {
            let n = 1 << i;

            let fixture = bench_fixture::big_struct_n(n);
            let (analysis, file_id) = fixture::file(&fixture);

            let time = Instant::now();

            let hash = analysis
                .highlight(HL_CONFIG, file_id)
                .unwrap()
                .iter()
                .filter(|it| it.highlight.tag == HlTag::Symbol(SymbolKind::Struct))
                .count();
            assert!(hash > n as usize);

            let elapsed = time.elapsed();
            al.sample(n as f64, elapsed.as_millis() as f64);
        }
    }
}

#[test]
fn benchmark_syntax_highlighting_parser() {
    if skip_slow_tests() {
        return;
    }

    let fixture = bench_fixture::glorious_old_parser();
    let (analysis, file_id) = fixture::file(&fixture);

    let hash = {
        let _pt = bench("syntax highlighting parser");
        analysis
            .highlight(HL_CONFIG, file_id)
            .unwrap()
            .iter()
            .filter(|it| {
                matches!(it.highlight.tag, HlTag::Symbol(SymbolKind::Function | SymbolKind::Method))
            })
            .count()
    };
    assert_eq!(hash, 1606);
}

#[test]
fn highlight_trait_with_lifetimes_regression_16958() {
    let (analysis, file_id) = fixture::file(
        r#"
pub trait Deserialize<'de> {
    fn deserialize();
}

fn f<'de, T: Deserialize<'de>>() {
    T::deserialize();
}
"#
        .trim(),
    );
    let _ = analysis.highlight(HL_CONFIG, file_id).unwrap();
}

#[test]
fn test_asm_highlighting() {
    check_highlighting(
        r#"
//- minicore: asm, concat
fn main() {
    unsafe {
        let foo = 1;
        let mut o = 0;
        core::arch::asm!(
            "%input = OpLoad _ {0}",
            concat!("%result = ", "bar", " _ %input"),
            "OpStore {1} %result",
            in(reg) &foo,
            in(reg) &mut o,
        );

        let thread_id: usize;
        core::arch::asm!("
            mov {0}, gs:[0x30]
            mov {0}, [{0}+0x48]
        ", out(reg) thread_id, options(pure, readonly, nostack));

        static UNMAP_BASE: usize;
        const MEM_RELEASE: usize;
        static VirtualFree: usize;
        const OffPtr: usize;
        const OffFn: usize;
        core::arch::asm!("
            push {free_type}
            push {free_size}
            push {base}

            mov eax, fs:[30h]
            mov eax, [eax+8h]
            add eax, {off_fn}
            mov [eax-{off_fn}+{off_ptr}], eax

            push eax

            jmp {virtual_free}
            ",
            off_ptr = const OffPtr,
            off_fn  = const OffFn,

            free_size = const 0,
            free_type = const MEM_RELEASE,

            virtual_free = sym VirtualFree,

            base = sym UNMAP_BASE,
            options(noreturn),
        );
    }
}
// taken from https://github.com/rust-embedded/cortex-m/blob/47921b51f8b960344fcfa1255a50a0d19efcde6d/cortex-m/src/asm.rs#L254-L274
#[inline]
pub unsafe fn bootstrap(msp: *const u32, rv: *const u32) -> ! {
    // Ensure thumb mode is set.
    let rv = (rv as u32) | 1;
    let msp = msp as u32;
    core::arch::asm!(
        "mrs {tmp}, CONTROL",
        "bics {tmp}, {spsel}",
        "msr CONTROL, {tmp}",
        "isb",
        "msr MSP, {msp}",
        "bx {rv}",
        // `out(reg) _` is not permitted in a `noreturn` asm! call,
        // so instead use `in(reg) 0` and don't restore it afterwards.
        tmp = in(reg) 0,
        spsel = in(reg) 2,
        msp = in(reg) msp,
        rv = in(reg) rv,
        options(noreturn, nomem, nostack),
    );
}
"#,
        expect_file!["./test_data/highlight_asm.html"],
        false,
    );
}

#[test]
fn issue_18089() {
    check_highlighting(
        r#"
//- proc_macros: issue_18089
fn main() {
    template!(template);
}

#[proc_macros::issue_18089]
fn template() {}
"#,
        expect_file!["./test_data/highlight_issue_18089.html"],
        false,
    );
}

#[test]
fn issue_19357() {
    check_highlighting(
        r#"
//- /foo.rs
fn main() {
    let x = &raw mut 5;
}
//- /main.rs
"#,
        expect_file!["./test_data/highlight_issue_19357.html"],
        false,
    );
}
