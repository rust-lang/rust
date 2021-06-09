#![crate_name = "foo"]
#![feature(const_generics_defaults)]

// @has foo/struct.Foo.html '//pre[@class="rust struct"]' \
//      'pub struct Foo<const M: usize = 10_usize, const N: usize = M, T = i32>(_);'
pub struct Foo<const M: usize = 10, const N: usize = M, T = i32>(T);
