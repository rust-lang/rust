// compile-flags: -Z parse-only

#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]

struct S<T: ?const ?const Tr>;
//~^ ERROR expected identifier, found keyword `const`
//~| ERROR expected one of `(`, `+`, `,`, `::`, `<`, `=`, or `>`
