//@ aux-build:all-item-types.rs
//@ build-aux-docs

#![crate_name = "foo"]

#[macro_use]
extern crate all_item_types;

//@ has 'foo/index.html' '//a[@href="../all_item_types/foo_mod/index.html"]' 'foo_mod'
#[doc(no_inline)]
pub use all_item_types::foo_mod;

//@ has 'foo/index.html' '//a[@href="../all_item_types/fn.foo_ffn.html"]' 'foo_ffn'
#[doc(no_inline)]
pub use all_item_types::foo_ffn;

//@ has 'foo/index.html' '//a[@href="../all_item_types/static.FOO_FSTATIC.html"]' 'FOO_FSTATIC'
#[doc(no_inline)]
pub use all_item_types::FOO_FSTATIC;

//@ has 'foo/index.html' '//a[@href="../all_item_types/foreigntype.FooFType.html"]' 'FooFType'
#[doc(no_inline)]
pub use all_item_types::FooFType;

//@ has 'foo/index.html' '//a[@href="../all_item_types/fn.foo_fn.html"]' 'foo_fn'
#[doc(no_inline)]
pub use all_item_types::foo_fn;

//@ has 'foo/index.html' '//a[@href="../all_item_types/trait.FooTrait.html"]' 'FooTrait'
#[doc(no_inline)]
pub use all_item_types::FooTrait;

//@ has 'foo/index.html' '//a[@href="../all_item_types/struct.FooStruct.html"]' 'FooStruct'
#[doc(no_inline)]
pub use all_item_types::FooStruct;

//@ has 'foo/index.html' '//a[@href="../all_item_types/enum.FooEnum.html"]' 'FooEnum'
#[doc(no_inline)]
pub use all_item_types::FooEnum;

//@ has 'foo/index.html' '//a[@href="../all_item_types/union.FooUnion.html"]' 'FooUnion'
#[doc(no_inline)]
pub use all_item_types::FooUnion;

//@ has 'foo/index.html' '//a[@href="../all_item_types/type.FooType.html"]' 'FooType'
#[doc(no_inline)]
pub use all_item_types::FooType;

//@ has 'foo/index.html' '//a[@href="../all_item_types/static.FOO_STATIC.html"]' 'FOO_STATIC'
#[doc(no_inline)]
pub use all_item_types::FOO_STATIC;

//@ has 'foo/index.html' '//a[@href="../all_item_types/constant.FOO_CONSTANT.html"]' 'FOO_CONSTANT'
#[doc(no_inline)]
pub use all_item_types::FOO_CONSTANT;

//@ has 'foo/index.html' '//a[@href="../all_item_types/macro.foo_macro.html"]' 'foo_macro'
#[doc(no_inline)]
pub use all_item_types::foo_macro;
