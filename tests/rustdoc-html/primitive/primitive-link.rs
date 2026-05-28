#![crate_name = "foo"]


//@ has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="{{channel}}/std/primitive.u32.html"]' 'u32'
//@ has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="{{channel}}/std/primitive.i64.html"]' 'i64'
//@ has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="{{channel}}/std/primitive.i32.html"]' 'std::primitive::i32'
//@ has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="{{channel}}/std/primitive.str.html"]' 'std::primitive::str'

//@ has foo/struct.Foo.html '//*[@class="docblock"]/p/a[@href="{{channel}}/std/primitive.i32.html#associatedconstant.MAX"]' 'std::primitive::i32::MAX'

/// It contains [`u32`] and [i64].
/// It also links to [std::primitive::i32], [std::primitive::str],
/// and [`std::primitive::i32::MAX`].
pub struct Foo;
