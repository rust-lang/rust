// compile-flags: -Z parse-only

#![feature(const_trait_impl)]

struct S<T: const Tr>;
//~^ ERROR expected identifier, found keyword `const`
//~| ERROR expected one of `(`, `+`, `,`, `::`, `<`, `=`, or `>`, found `Tr`
