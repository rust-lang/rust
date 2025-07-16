#![crate_name = "allow_unsized"]

use std::fmt::Debug;
use std::marker::PhantomData;

pub trait CustomSized: Sized {}
impl CustomSized for u8 {}


// Generic functions
//@ !has allow_unsized/fn.func_custom.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_custom<T: ?Sized + CustomSized>() {}

//@ !has allow_unsized/fn.func_custom_where_denies.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_custom_where_denies<T: ?Sized>() where T: CustomSized {}

//@ !has allow_unsized/fn.func_custom_where_allows.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_custom_where_allows<T: CustomSized>() where T: ?Sized {}

//@ !has allow_unsized/fn.func_custom_where_both.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_custom_where_both<T>() where T: ?Sized + CustomSized {}

//@ has allow_unsized/fn.func_unsized.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_unsized<T: ?Sized>() {}

//@ !has allow_unsized/fn.func_clone.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_clone<T: ?Sized + Clone>() {}

//@ has allow_unsized/fn.func_debug.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn func_debug<T: ?Sized + Debug>() {}

// Generic structs
//@ !has allow_unsized/struct.StructCustom.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct StructCustom<T: ?Sized + CustomSized>(PhantomData<T>);

//@ has allow_unsized/struct.StructUnsized.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct StructUnsized<T: ?Sized>(PhantomData<T>);

//@ !has allow_unsized/struct.StructClone.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct StructClone<T: ?Sized + Clone>(PhantomData<T>);

//@ has allow_unsized/struct.StructDebug.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct StructDebug<T: ?Sized + Debug>(PhantomData<T>);


// Structs with `?Sized` bound, and impl blocks that add additional bounds.
// The structs have to be different due to limitations in the XPath matching syntax.
//@ has allow_unsized/struct.CustomSizedWrapper.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct CustomSizedWrapper<T: ?Sized>(PhantomData<T>);

//@ !has allow_unsized/struct.CustomSizedWrapper.html "//div[@id='implementations-list']//section[@class='impl']//h3" '?Sized'
impl<T: ?Sized + CustomSized> CustomSizedWrapper<T> {
    pub fn impl_custom() {}
}

//@ has allow_unsized/struct.CloneWrapper.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct CloneWrapper<T: ?Sized>(PhantomData<T>);

//@ !has allow_unsized/struct.CloneWrapper.html "//div[@id='implementations-list']//section[@class='impl']//h3" '?Sized'
impl<T: ?Sized + Clone> CloneWrapper<T> {
    pub fn impl_clone() {}
}

//@ has allow_unsized/struct.DebugWrapper.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct DebugWrapper<T: ?Sized>(PhantomData<T>);

//@ has allow_unsized/struct.DebugWrapper.html "//div[@id='implementations-list']//section[@class='impl']//h3" '?Sized'
impl<T: ?Sized + Debug> DebugWrapper<T> {
    pub fn impl_debug() {}
}

//@ has allow_unsized/struct.Wrapper.html '//pre[@class="rust item-decl"]' '?Sized'
pub struct Wrapper<T: ?Sized>(PhantomData<T>);

impl<T: ?Sized> Wrapper<T> {
    //@ !has allow_unsized/struct.Wrapper.html "//*[@id='method.assoc_custom']"//h4 '?Sized'
    pub fn assoc_custom<U: ?Sized + CustomSized>(&self) {}

    //@ has allow_unsized/struct.Wrapper.html "//*[@id='method.assoc_unsized']"//h4 '?Sized'
    pub fn assoc_unsized<U: ?Sized>(&self) {}

    //@ !has allow_unsized/struct.Wrapper.html "//*[@id='method.assoc_clone']"//h4 '?Sized'
    pub fn assoc_clone<U: ?Sized + Clone>(&self) {}

    //@ has allow_unsized/struct.Wrapper.html "//*[@id='method.assoc_debug']"//h4 '?Sized'
    pub fn assoc_debug<U: ?Sized + Debug>(&self) {}
}


// Traits with generic parameters
//@ !has allow_unsized/trait.TraitCustom.html '//pre[@class="rust item-decl"]' '?Sized'
pub trait TraitCustom<T: ?Sized + CustomSized> {}

//@ has allow_unsized/trait.TraitUnsized.html '//pre[@class="rust item-decl"]' '?Sized'
pub trait TraitUnsized<T: ?Sized> {}

//@ !has allow_unsized/trait.TraitClone.html '//pre[@class="rust item-decl"]' '?Sized'
pub trait TraitClone<T: ?Sized + Clone> {}

//@ has allow_unsized/trait.TraitDebug.html '//pre[@class="rust item-decl"]' '?Sized'
pub trait TraitDebug<T: ?Sized + Debug> {}

pub trait TraitMethods {
    //@ !has allow_unsized/trait.TraitMethods.html "//*[@id='tymethod.method_custom']"//h4 '?Sized'
    fn method_custom<T: ?Sized + CustomSized>();

    //@ has allow_unsized/trait.TraitMethods.html "//*[@id='tymethod.method_unsized']"//h4 '?Sized'
    fn method_unsized<T: ?Sized>();

    //@ !has allow_unsized/trait.TraitMethods.html "//*[@id='tymethod.method_clone']"//h4 '?Sized'
    fn method_clone<T: ?Sized + Clone>();

    //@ has allow_unsized/trait.TraitMethods.html "//*[@id='tymethod.method_debug']"//h4 '?Sized'
    fn method_debug<T: ?Sized + Debug>();
}


// `where` clauses on trait functions, which only affect `T` for that function
//@ has allow_unsized/trait.OuterDebug.html '//pre[@class="rust item-decl"]' '?Sized'
pub trait OuterDebug<T: ?Sized> {
    fn foo() where T: Debug;
}

//@ has allow_unsized/trait.OuterClone.html '//pre[@class="rust item-decl"]' '?Sized'
pub trait OuterClone<T: ?Sized> {
    fn foo() where T: Clone;
}

//@ !has allow_unsized/fn.synth_clone.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn synth_clone(_: impl Clone + ?Sized) {}

//@ has allow_unsized/fn.synth_debug.html '//pre[@class="rust item-decl"]' '?Sized'
pub fn synth_debug(_: impl Debug + ?Sized) {}
