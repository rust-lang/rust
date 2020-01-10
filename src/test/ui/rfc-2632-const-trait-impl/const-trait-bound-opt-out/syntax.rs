// compile-flags: -Z parse-only
// check-pass

#![feature(const_trait_bound_opt_out)]
#![allow(incomplete_features)]

struct S<
    T: ?const ?for<'a> Tr<'a> + 'static + ?const std::ops::Add,
    T: ?const ?for<'a: 'b> m::Trait<'a>,
>;
