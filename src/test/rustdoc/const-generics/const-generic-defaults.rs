#![crate_name = "foo"]

// @has foo/struct.Foo.html '//pre[@class="rust struct"]' \
//      'pub struct Foo<const M: usize = 10_usize, const N: usize = M, T = i32>(_);'
pub struct Foo<const M: usize = 10, const N: usize = M, T = i32>(T);
