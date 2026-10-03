#![feature(generic_const_items)]
#![allow(incomplete_features)]

// (Here we hide the RHS because it's not a literal)
//
//@ has 'generic_const_items/constant.K.html'
//@ has - '//*[@class="rust item-decl"]//code' \
// "pub const K<'a, T: 'a + Copy, const N: usize>: Option<[T; N]> \
// where \
//     String: From<T>;"
pub const K<'a, T: 'a + Copy, const N: usize>: Option<[T; N]> = None
where
    String: From<T>;

// Ensure that we render the where-clause *after* the RHS.
//
//@ has 'generic_const_items/constant.Q.html'
//@ has - '//*[@class="rust item-decl"]//code' \
// "pub const Q: i32 = 0 \
// where \
//     for<'_delay> i32: Copy;"
pub const Q: i32 = 0 where for<'_delay> i32: Copy;

// Ensure that we properly insert the fake comment containing the value
// between the expression and the where-clause.
//
//@ has 'generic_const_items/constant.C.html'
//@ has - '//*[@class="rust item-decl"]//code' \
// "pub const C<'a>: u8 = _ /* 2u8 */ \
// where \
//     &'a (): Sized;"
pub const C<'a>: u8 = 1 + 1 where &'a (): Sized;

//@ has generic_const_items/trait.Trait.html
pub trait Trait<T: ?Sized> {
    //@ has - '//*[@id="associatedconstant.C"]' \
    // "const C<'a>: &'a T \
    // where \
    //     T: 'a + Eq"
    const C<'a>: &'a T
    where
        T: 'a + Eq;
}

pub struct Implementor;

//@ has generic_const_items/struct.Implementor.html
//@ has - '//h3[@class="code-header"]' 'impl Trait<str> for Implementor'
impl Trait<str> for Implementor {
    //@ has - '//*[@id="associatedconstant.C"]' \
    // "const C<'a>: &'a str = \"C\" \
    // where \
    //     str: 'a"
    const C<'a>: &'a str = "C"
    // In real code we could've left off this bound but adding it explicitly allows us to test if
    // we render where-clauses on associated consts inside impl blocks correctly.
    where
        str: 'a;
}
