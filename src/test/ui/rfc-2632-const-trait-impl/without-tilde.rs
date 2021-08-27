// compiler-flags: -Z parse-only

#![feature(const_trait_impl)]

struct S<T: const Tr>;
//~^ ERROR expected one of `!`, `(`, `,`, `=`, `>`, `?`, `for`, `~`, lifetime, or path
