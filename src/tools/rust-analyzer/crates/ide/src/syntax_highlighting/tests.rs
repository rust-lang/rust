use std::time::Instant;

use expect_test::{expect_file, ExpectFile};
use ide_db::SymbolKind;
use test_utils::{bench, bench_fixture, skip_slow_tests, AssertLinear};

use crate::{fixture, FileRange, HighlightConfig, HlTag, TextRange};

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
//- minicore: derive, copy
#[allow(dead_code)]
#[rustfmt::skip]
#[proc_macros::identity]
#[derive(Copy)]
/// This is a doc comment
// This is a normal comment
/// This is a doc comment
#[derive(Copy)]
// This is another normal comment
/// This is another doc comment
// This is another normal comment
#[derive(Copy)]
// The reason for these being here is to test AttrIds
struct Foo;
"#,
        expect_file!["./test_data/highlight_attributes.html"],
        false,
    );
}

#[test]
fn macros() {
    check_highlighting(
        r#"
//- proc_macros: mirror
proc_macros::mirror! {
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

fn main() {
    println!("Hello, {}!", 92);
    dont_color_me_braces!();
    noop!(noop!(1));
}
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
//- minicore: derive, copy
//- /main.rs crate:main deps:foo
use inner::{self as inner_mod};
mod inner {}

pub mod ops {
    #[lang = "fn_once"]
    pub trait FnOnce<Args> {}

    #[lang = "fn_mut"]
    pub trait FnMut<Args>: FnOnce<Args> {}

    #[lang = "fn"]
    pub trait Fn<Args>: FnMut<Args> {}
}

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

use ops::Fn;
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
const USAGE_OF_BOOL:bool = Bool::True.to_primitive();

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
    check_highlighting(
        r#"
extern crate self;

use crate;
use self;
mod __ {
    use super::*;
}

macro_rules! void {
    ($($tt:tt)*) => {}
}
void!(Self);
struct __ where Self:;
fn __(_: Self) {}
"#,
        expect_file!["./test_data/highlight_keywords.html"],
        false,
    );
}

#[test]
fn test_string_highlighting() {
    // The format string detection is based on macro-expansion,
    // thus, we have to copy the macro definition from `std`
    check_highlighting(
        r#"
macro_rules! println {
    ($($arg:tt)*) => ({
        $crate::io::_print($crate::format_args_nl!($($arg)*));
    })
}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! format_args {}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! const_format_args {}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! format_args_nl {}

mod panic {
    pub macro panic_2015 {
        () => (
            $crate::panicking::panic("explicit panic")
        ),
        ($msg:literal $(,)?) => (
            $crate::panicking::panic($msg)
        ),
        // Use `panic_str` instead of `panic_display::<&str>` for non_fmt_panic lint.
        ($msg:expr $(,)?) => (
            $crate::panicking::panic_str($msg)
        ),
        // Special-case the single-argument case for const_panic.
        ("{}", $arg:expr $(,)?) => (
            $crate::panicking::panic_display(&$arg)
        ),
        ($fmt:expr, $($arg:tt)+) => (
            $crate::panicking::panic_fmt($crate::const_format_args!($fmt, $($arg)+))
        ),
    }
}

#[rustc_builtin_macro(std_panic)]
#[macro_export]
macro_rules! panic {}
#[rustc_builtin_macro]
macro_rules! assert {}
#[rustc_builtin_macro]
macro_rules! asm {}

macro_rules! toho {
    () => ($crate::panic!("not yet implemented"));
    ($($arg:tt)+) => ($crate::panic!("not yet implemented: {}", $crate::format_args!($($arg)+)));
}

fn main() {
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

    let _ = "\x28\x28\x00\x63\n";
    let _ = b"\x28\x28\x00\x63\n";

    println!("{\x41}", A = 92);
    println!("{ничоси}", ничоси = 92);

    println!("{:x?} {} ", thingy, n2);
    panic!("{}", 0);
    panic!("more {}", 1);
    assert!(true, "{}", 1);
    assert!(true, "{} asdasd", 1);
    toho!("{}fmt", 0);
    asm!("mov eax, {0}");
    format_args!(concat!("{}"), "{}");
}"#,
        expect_file!["./test_data/highlight_strings.html"],
        false,
    );
}

#[test]
fn test_unsafe_highlighting() {
    check_highlighting(
        r#"
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
static mut MUT_GLOBAL: Struct = Struct { field: 0 };
static GLOBAL: Struct = Struct { field: 0 };
unsafe fn unsafe_fn() {}

union Union {
    a: u32,
    b: f32,
}

struct Struct { field: i32 }
impl Struct {
    unsafe fn unsafe_method(&self) {}
}

#[repr(packed)]
struct Packed {
    a: u16,
}

unsafe trait UnsafeTrait {}
unsafe impl UnsafeTrait for Packed {}
impl !UnsafeTrait for () {}

fn unsafe_trait_bound<T: UnsafeTrait>(_: T) {}

trait DoTheAutoref {
    fn calls_autoref(&self);
}

impl DoTheAutoref for u16 {
    fn calls_autoref(&self) {}
}

fn main() {
    let x = &5 as *const _ as *const usize;
    let u = Union { b: 0 };

    id! {
        unsafe { unsafe_deref!() }
    };

    unsafe {
        unsafe_deref!();
        id! { unsafe_deref!() };

        // unsafe fn and method calls
        unsafe_fn();
        let b = u.b;
        match u {
            Union { b: 0 } => (),
            Union { a } => (),
        }
        Struct { field: 0 }.unsafe_method();

        // unsafe deref
        *x;

        // unsafe access to a static mut
        MUT_GLOBAL.field;
        GLOBAL.field;

        // unsafe ref of packed fields
        let packed = Packed { a: 0 };
        let a = &packed.a;
        let ref a = packed.a;
        let Packed { ref a } = packed;
        let Packed { a: ref _a } = packed;

        // unsafe auto ref of packed field
        packed.a.calls_autoref();
    }
}
"#,
        expect_file!["./test_data/highlight_unsafe.html"],
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
//- /main.rs crate:main deps:std,alloc
extern crate std;
extern crate alloc as abc;
//- /std/lib.rs crate:std
pub struct S;
//- /alloc/lib.rs crate:alloc
pub struct A
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
fn fixture(ra_fixture: &str) {}

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

#[test]
#[cfg_attr(
    not(all(unix, target_pointer_width = "64")),
    ignore = "depends on `DefaultHasher` outputs"
)]
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

/// Highlights the code given by the `ra_fixture` argument, renders the
/// result as HTML, and compares it with the HTML file given as `snapshot`.
/// Note that the `snapshot` file is overwritten by the rendered HTML.
fn check_highlighting(ra_fixture: &str, expect: ExpectFile, rainbow: bool) {
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
            .filter(|it| it.highlight.tag == HlTag::Symbol(SymbolKind::Function))
            .count()
    };
    assert_eq!(hash, 1609);
}
