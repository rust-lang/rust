// compile-flags: -Z parse-only
// check-pass

#![feature(const_trait_impl)]

struct S<
    T: ~const ?for<'a> Tr<'a> + 'static + ~const std::ops::Add,
    T: ~const ?for<'a: 'b> m::Trait<'a>,
>;
