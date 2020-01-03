// compile-flags: -Z parse-only

#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]

struct S<T: const Tr>;
//~^ ERROR expected one of `!`, `(`, `,`, `=`, `>`, `?`, `for`, lifetime, or path
