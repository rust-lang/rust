// compile-flags: -Z parse-only

#![feature(const_trait_impl)]
#![feature(effects)]

struct S<T: const Tr>;
//~^ ERROR const bounds must start with `~`
