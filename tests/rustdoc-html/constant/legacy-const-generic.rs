#![crate_name = "foo"]
#![feature(rustc_attrs)]

//@ has 'foo/fn.foo.html'
//@ has - '//pre[@class="rust item-decl"]' 'fn foo(x: usize, const Y: usize, z: usize) -> [usize; 3]'
#[rustc_legacy_const_generics(1)]
pub fn foo<const Y: usize>(x: usize, z: usize) -> [usize; 3] {
    [x, Y, z]
}

//@ has 'foo/fn.bar.html'
//@ has - '//pre[@class="rust item-decl"]' 'fn bar(x: usize, const Y: usize, const Z: usize) -> [usize; 3]'
#[rustc_legacy_const_generics(1, 2)]
pub fn bar<const Y: usize, const Z: usize>(x: usize) -> [usize; 3] {
    [x, Y, z]
}
