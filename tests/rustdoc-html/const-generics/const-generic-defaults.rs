#![crate_name = "foo"]

//@ has foo/type.Foo.html '//pre[@class="rust item-decl"]' \
//      'pub struct Foo<const M: usize = 10, const N: usize = M, T = i32>('
pub struct Foo<const M: usize = 10, const N: usize = M, T = i32>(T);
