#![crate_name = "foo"]

// @has foo/fn.tuple0.html //pre 'pub fn tuple0(x: ())'
pub fn tuple0(x: ()) -> () { x }
// @has foo/fn.tuple1.html //pre 'pub fn tuple1(x: (i32,)) -> (i32,)'
pub fn tuple1(x: (i32,)) -> (i32,) { x }
// @has foo/fn.tuple2.html //pre 'pub fn tuple2(x: (i32, i32)) -> (i32, i32)'
pub fn tuple2(x: (i32, i32)) -> (i32, i32) { x }
