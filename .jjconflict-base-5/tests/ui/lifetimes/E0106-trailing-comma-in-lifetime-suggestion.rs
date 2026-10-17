// Regression test for <https://github.com/rust-lang/rust/issues/154600>.
//
// When suggesting lifetime parameters for empty angle brackets like `Foo<>`,
// the suggestion should not include a trailing comma (e.g., `Foo<'a>` not `Foo<'a, >`).
// When there are other generic arguments like `Foo<T>`, the trailing comma is needed
// (e.g., `Foo<'a, T>`).

#![crate_type = "lib"]

struct Foo<'a>(&'a ());

type A = Foo<>;
//~^ ERROR missing lifetime specifier [E0106]

struct Bar<'a, T>(&'a T);

type B = Bar<u8>;
//~^ ERROR missing lifetime specifier [E0106]
