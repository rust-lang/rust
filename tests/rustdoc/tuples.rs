#![crate_name = "foo"]

// @has foo/fn.tuple0.html //pre 'pub fn tuple0(x: ())'
// @snapshot link_unit - '//pre[@class="rust item-decl"]/code'
pub fn tuple0(x: ()) -> () { x }
// @has foo/fn.tuple1.html //pre 'pub fn tuple1(x: (i32,)) -> (i32,)'
// @snapshot link1_i32 - '//pre[@class="rust item-decl"]/code'
pub fn tuple1(x: (i32,)) -> (i32,) { x }
// @has foo/fn.tuple2.html //pre 'pub fn tuple2(x: (i32, i32)) -> (i32, i32)'
// @snapshot link2_i32 - '//pre[@class="rust item-decl"]/code'
pub fn tuple2(x: (i32, i32)) -> (i32, i32) { x }
// @has foo/fn.tuple1_t.html //pre 'pub fn tuple1_t<T>(x: (T,)) -> (T,)'
// @snapshot link1_t - '//pre[@class="rust item-decl"]/code'
pub fn tuple1_t<T>(x: (T,)) -> (T,) { x }
// @has foo/fn.tuple2_t.html //pre 'pub fn tuple2_t<T>(x: (T, T)) -> (T, T)'
// @snapshot link2_t - '//pre[@class="rust item-decl"]/code'
pub fn tuple2_t<T>(x: (T, T)) -> (T, T) { x }
// @has foo/fn.tuple2_tu.html //pre 'pub fn tuple2_tu<T, U>(x: (T, U)) -> (T, U)'
// @snapshot link2_tu - '//pre[@class="rust item-decl"]/code'
pub fn tuple2_tu<T, U>(x: (T, U)) -> (T, U) { x }
