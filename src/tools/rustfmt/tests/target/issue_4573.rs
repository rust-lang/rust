// rustmft-version:Two
// rustmft-use_small_heuristics:Max
// rustmft-merge_derives:false
// These are the same rustfmt configuration options that are used
// in the comiler as of ce39461ca75a and 8eb7c58dbb7b
// These are commits in https://github.com/rust-lang/rust

#![no_std] // inner attribute comment
// inner attribute comment
#![no_implicit_prelude]
// post inner attribute comment

#[cfg(not(miri))] // inline comment
#[no_link]
extern crate foo;

// before attributes
#[no_link]
// between attributes
#[cfg(not(miri))] // inline comment
extern crate foo as bar;

#[cfg(not(miri))] // inline comment
// between attribute and use
use foo;

#[cfg(not(miri))] // inline comment
use foo;

/* pre attributre */
#[cfg(not(miri))]
use foo::bar;

#[cfg(not(miri))] // inline comment
use foo::bar as FooBar;

#[cfg(not(miri))] // inline comment
#[allow(unused)]
#[deprecated(
    since = "5.2",  // inline inner comment
    note = "FOO was rarely used. Users should instead use BAR"
)]
#[allow(unused)]
static FOO: i32 = 42;

#[used]
#[export_name = "FOO"]
#[cfg(not(miri))] // inline comment
#[deprecated(
    since = "5.2",
    note = "FOO was rarely used. Users should instead use BAR"
)]
static FOO: i32 = 42;

#[cfg(not(miri))] // inline comment
#[export_name = "FOO"]
static BAR: &'static str = "bar";

#[cfg(not(miri))] // inline comment
const BAR: i32 = 42;

#[cfg(not(miri))] // inline comment
#[no_mangle]
#[link_section = ".example_section"]
fn foo(bar: usize) {
    #[cfg(not(miri))] // inline comment
    println!("hello world!");
}

#[cfg(not(miri))] // inline comment
mod foo {}

#[cfg(not(miri))] // inline comment
extern "C" {
    fn my_c_function(x: i32) -> bool;
}

#[cfg(not(miri))] // inline comment
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {

    #[link_name = "actual_symbol_name"] // inline comment
    // between attribute and function
    fn my_c_function(x: i32) -> bool;
}

#[cfg(not(miri))] // inline comment
pub extern "C" fn callable_from_c(x: i32) -> bool {
    x % 3 == 0
}

#[cfg(not(miri))] // inline comment
/* between attribute block comment */
#[no_mangle]
/* between attribute and type */
type Foo = Bar<u8>;

#[no_mangle]
#[cfg(not(miri))] // inline comment
#[non_exhaustive] // inline comment
enum Foo {
    Bar,
    Baz,
}

#[no_mangle]
#[cfg(not(miri))] /* inline comment */
struct Foo<A> {
    x: A,
}

#[cfg(not(miri))] // inline comment
union Foo<A, B> {
    x: A,
    y: B,
}

#[cfg(not(miri))] // inline comment
trait Foo {}

#[cfg(not(miri))] // inline comment
trait Foo = Bar + Quux;

#[cfg(not(miri))] // inline comment
impl Foo {}

#[cfg(not(miri))] // inline comment
macro_rules! bar {
    (3) => {};
}

mod nested {
    #[cfg(not(miri))] // inline comment
    // between attribute and use
    use foo;

    #[cfg(not(miri))] // inline comment
    use foo;

    #[cfg(not(miri))] // inline comment
    use foo::bar;

    #[cfg(not(miri))] // inline comment
    use foo::bar as FooBar;

    #[cfg(not(miri))] // inline comment
    static FOO: i32 = 42;

    #[cfg(not(miri))] // inline comment
    static FOO: i32 = 42;

    #[cfg(not(miri))] // inline comment
    static FOO: &'static str = "bar";

    #[cfg(not(miri))] // inline comment
    const FOO: i32 = 42;

    #[cfg(not(miri))] // inline comment
    fn foo(bar: usize) {
        #[cfg(not(miri))] // inline comment
        println!("hello world!");
    }

    #[cfg(not(miri))] // inline comment
    mod foo {}

    #[cfg(not(miri))] // inline comment
    mod foo {}

    #[cfg(not(miri))] // inline comment
    extern "C" {
        fn my_c_function(x: i32) -> bool;
    }

    #[cfg(not(miri))] // inline comment
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {

        #[link_name = "actual_symbol_name"] // inline comment
        // between attribute and function
        fn my_c_function(x: i32) -> bool;
    }

    #[cfg(not(miri))] // inline comment
    pub extern "C" fn callable_from_c(x: i32) -> bool {
        x % 3 == 0
    }

    #[cfg(not(miri))] // inline comment
    type Foo = Bar<u8>;

    #[cfg(not(miri))] // inline comment
    #[non_exhaustive] // inline comment
    enum Foo {
        // comment
        #[attribute_1]
        #[attribute_2] // comment
        // comment!
        Bar,
        /* comment */
        #[attribute_1]
        #[attribute_2] /* comment */
        #[attribute_3]
        #[attribute_4]
        /* comment! */
        Baz,
    }

    #[cfg(not(miri))] // inline comment
    struct Foo<A> {
        x: A,
    }

    #[cfg(not(miri))] // inline comment
    union Foo<A, B> {
        #[attribute_1]
        #[attribute_2] /* comment */
        #[attribute_3]
        #[attribute_4] // comment
        x: A,
        y: B,
    }

    #[cfg(not(miri))] // inline comment
    #[allow(missing_docs)]
    trait Foo {
        #[must_use] /* comment
                     * that wrappes to
                     * the next line */
        fn bar() {}
    }

    #[allow(missing_docs)]
    #[cfg(not(miri))] // inline comment
    trait Foo = Bar + Quux;

    #[allow(missing_docs)]
    #[cfg(not(miri))] // inline comment
    impl Foo {}

    #[cfg(not(miri))] // inline comment
    macro_rules! bar {
        (3) => {};
    }
}
