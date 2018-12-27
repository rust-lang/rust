// aux-build:assoc-items.rs
// build-aux-docs
// ignore-cross-compile

#![crate_name = "foo"]

extern crate assoc_items;

// @has foo/struct.MyStruct.html
// @!has - 'PrivateConst'
// @has - '//*[@id="associatedconstant.PublicConst"]' 'pub const PublicConst: u8'
// @has - '//*[@class="docblock"]' 'docs for PublicConst'
// @!has - 'private_method'
// @has - '//*[@id="method.public_method"]' 'pub fn public_method()'
// @has - '//*[@class="docblock"]' 'docs for public_method'
// @has - '//*[@id="associatedconstant.ConstNoDefault"]' 'const ConstNoDefault: i16'
// @has - '//*[@class="docblock"]' 'dox for ConstNoDefault'
// @has - '//*[@id="associatedconstant.ConstWithDefault"]' 'const ConstWithDefault: u16'
// @has - '//*[@class="docblock"]' 'docs for ConstWithDefault'
// @has - '//*[@id="associatedtype.TypeNoDefault"]' 'type TypeNoDefault = i32'
// @has - '//*[@class="docblock"]' 'dox for TypeNoDefault'
// @has - '//*[@id="associatedtype.TypeWithDefault"]' 'type TypeWithDefault = u32'
// @has - '//*[@class="docblock"]' 'docs for TypeWithDefault'
// @has - '//*[@id="method.method_no_default"]' 'fn method_no_default()'
// @has - '//*[@class="docblock"]' 'dox for method_no_default'
// @has - '//*[@id="method.method_with_default"]' 'fn method_with_default()'
// @has - '//*[@class="docblock"]' 'docs for method_with_default'
pub use assoc_items::MyStruct;

// @has foo/trait.MyTrait.html
// @has - '//*[@id="associatedconstant.ConstNoDefault"]' 'const ConstNoDefault: i16'
// @has - '//*[@class="docblock"]' 'docs for ConstNoDefault'
// @has - '//*[@id="associatedconstant.ConstWithDefault"]' 'const ConstWithDefault: u16'
// @has - '//*[@class="docblock"]' 'docs for ConstWithDefault'
// @has - '//*[@id="associatedtype.TypeNoDefault"]' 'type TypeNoDefault'
// @has - '//*[@class="docblock"]' 'docs for TypeNoDefault'
// @has - '//*[@class="docblock"]' 'docs for TypeWithDefault'
// @has - '//*[@id="tymethod.method_no_default"]' 'fn method_no_default()'
// @has - '//*[@class="docblock"]' 'docs for method_no_default'
// @has - '//*[@id="method.method_with_default"]' 'fn method_with_default()'
// @has - '//*[@class="docblock"]' 'docs for method_with_default'
pub use assoc_items::MyTrait;
