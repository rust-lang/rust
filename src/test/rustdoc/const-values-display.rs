#![crate_name = "foo"]

pub trait Test {}
pub struct Array<T, const N: usize>([T; N]);

// @has 'foo/trait.Test.html'
// @has - '//*[@class="code-header in-band"]' 'impl Test for [u8; 2]'
impl Test for [u8; 1 + 1] {}
// @has - '//*[@class="code-header in-band"]' 'impl Test for Array<u8, 2>'
impl Test for Array<u8, {1 + 1}> {}

type ArrayType<T, const N: usize> = [T; N];

// @has 'foo/type.A0.html'
// @has - '//*[@class="docblock item-decl"]' 'pub type A0 = [i32; 5];'
pub type A0 = [i32; 3 + 2];
// @has 'foo/type.A1.html'
// @has - '//*[@class="docblock item-decl"]' 'pub type A1 = Array<i32, 5>;'
pub type A1 = Array<i32, { 3 + 2 }>;
// @has 'foo/struct.B0.html'
// @has - '//*[@class="docblock item-decl"]' 'pub struct B0(pub [i32; 6]);'
// @has - '//*[@id="structfield.0"]' '0: [i32; 6]'
pub struct B0(pub [i32; 3 * 2]);
// @has 'foo/struct.B1.html'
// @has - '//*[@class="docblock item-decl"]' 'pub struct B1(pub Array<i32, 5>);'
// @has - '//*[@id="structfield.0"]' '0: Array<i32, 5>'
pub struct B1(pub Array<i32, {3 + 2}>);
// @has 'foo/struct.C0.html'
// @has - '//*[@class="docblock item-decl"]' 'pub foo: [i32; 6],'
// @has - '//*[@id="structfield.foo"]' 'foo: [i32; 6]'
pub struct C0 { pub foo: [i32; 3 * 2] }
// @has 'foo/struct.C1.html'
// @has - '//*[@class="docblock item-decl"]' 'pub foo: Array<i32, 5>,'
// @has - '//*[@id="structfield.foo"]' 'foo: Array<i32, 5>'
pub struct C1 { pub foo: Array<i32, {3 + 2}> }

// @has 'foo/constant.X.html'
// @has - '//*[@class="docblock item-decl"]' 'pub const X: u32 = 14 * 2; // 28u32'
pub const X: u32 = 14 * 2;
