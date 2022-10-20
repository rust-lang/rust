// compile-flags: -Z parse-only

#![feature(const_trait_impl)]
#![feature(effects)]

struct S<T: ~const ~const Tr>;
//~^ ERROR expected identifier, found `~`
