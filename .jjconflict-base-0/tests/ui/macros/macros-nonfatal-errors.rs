//@ normalize-stderr: "`: .*" -> "`: $$FILE_NOT_FOUND_MSG"

// test that errors in a (selection) of macros don't kill compilation
// immediately, so that we get more errors listed at a time.

#![feature(trace_macros)]
#![feature(stmt_expr_attributes)]

use std::arch::asm;

#[derive(Default)]
struct DefaultInnerAttrStruct {
    #[default] //~ ERROR the `#[default]` attribute may only be used on unit enum variants
    foo: (),
}

#[derive(Default)]
struct DefaultInnerAttrTupleStruct(#[default] ());
//~^ ERROR the `#[default]` attribute may only be used on unit enum variants

#[derive(Default)]
#[default] //~ ERROR the `#[default]` attribute may only be used on unit enum variants
struct DefaultOuterAttrStruct {}

#[derive(Default)]
#[default] //~ ERROR the `#[default]` attribute may only be used on unit enum variants
enum DefaultOuterAttrEnum {
    #[default]
    Foo,
}

#[rustfmt::skip] // needs some work to handle this case
#[repr(u8)]
#[derive(Default)]
enum AttrOnInnerExpression {
    Foo = #[default] 0, //~ ERROR the `#[default]` attribute may only be used on unit enum variants
    Bar([u8; #[default] 1]), //~ ERROR the `#[default]` attribute may only be used on unit enum variants
    #[default]
    Baz,
}

#[derive(Default)] //~ ERROR `#[derive(Default)]` on enum with no `#[default]`
enum NoDeclaredDefault {
    Foo,
    Bar,
}

#[derive(Default)] //~ ERROR `#[derive(Default)]` on enum with no `#[default]`
enum NoDeclaredDefaultWithoutUnitVariant {
    Foo(i32),
    Bar(i32),
}

#[derive(Default)] //~ ERROR multiple declared defaults
enum MultipleDefaults {
    #[default]
    Foo,
    #[default]
    Bar,
    #[default]
    Baz,
}

#[derive(Default)]
enum ExtraDeriveTokens {
    #[default = 1] //~ ERROR `#[default]` attribute does not accept a value
    Foo,
}

#[derive(Default)]
enum TwoDefaultAttrs {
    #[default]
    #[default]
    Foo, //~ERROR multiple `#[default]` attributes
    Bar,
}

#[derive(Default)]
enum ManyDefaultAttrs {
    #[default]
    #[default]
    #[default]
    #[default]
    Foo, //~ERROR multiple `#[default]` attributes
    Bar,
}

#[derive(Default)]
enum DefaultHasFields {
    #[default]
    Foo {}, //~ ERROR the `#[default]` attribute may only be used on unit enum variants
    Bar,
}

#[derive(Default)]
enum NonExhaustiveDefault {
    #[default]
    #[non_exhaustive]
    Foo, //~ ERROR default variant must be exhaustive
    Bar,
}

fn main() {
    asm!(invalid); //~ ERROR
    llvm_asm!(invalid); //~ ERROR

    option_env!(invalid); //~ ERROR
    env!(invalid); //~ ERROR
    env!(foo, abr, baz); //~ ERROR
    env!("RUST_HOPEFULLY_THIS_DOESNT_EXIST"); //~ ERROR

    format!(invalid); //~ ERROR

    include!(invalid); //~ ERROR

    include_str!(invalid); //~ ERROR
    include_str!("i'd be quite surprised if a file with this name existed"); //~ ERROR
    include_bytes!(invalid); //~ ERROR
    include_bytes!("i'd be quite surprised if a file with this name existed"); //~ ERROR

    trace_macros!(invalid); //~ ERROR
}

/// Check that `#[derive(Default)]` does use a `T : Default` bound when the
/// `#[default]` variant is `#[non_exhaustive]` (should this end up allowed).
const _: () = {
    #[derive(Default)]
    enum NonExhaustiveDefaultGeneric<T> {
        #[default]
        #[non_exhaustive]
        Foo, //~ ERROR default variant must be exhaustive
        Bar(T),
    }

    fn assert_impls_default<T: Default>() {}

    enum NotDefault {}

    // Note: the `derive(Default)` currently bails early enough for trait-checking
    // not to happen. Should it bail late enough, or even pass, make sure to
    // assert that the following line fails.
    let _ = assert_impls_default::<NonExhaustiveDefaultGeneric<NotDefault>>;
};
