#![crate_name = "allow_unsized"]

use std::fmt::Debug;
use std::marker::PhantomData;

pub trait CustomSized: Sized {}
impl CustomSized for u8 {}

// Generic functions
//@ is "$.index[?(@.name=='func_custom')].inner.function.generics.params[0].kind.type.allow_unsized" false
pub fn func_custom<T: ?Sized + CustomSized>() {}

//@ is "$.index[?(@.name=='func_custom')].inner.function.generics.params[0].kind.type.allow_unsized" false
pub fn func_custom_where_denies<T: ?Sized>()
where
    T: CustomSized,
{
}

//@ is "$.index[?(@.name=='func_custom')].inner.function.generics.params[0].kind.type.allow_unsized" false
pub fn func_custom_where_allows<T: CustomSized>()
where
    T: ?Sized,
{
}

//@ is "$.index[?(@.name=='func_custom')].inner.function.generics.params[0].kind.type.allow_unsized" false
pub fn func_custom_where_both<T>()
where
    T: ?Sized + CustomSized,
{
}

//@ is "$.index[?(@.name=='func_unsized')].inner.function.generics.params[0].kind.type.allow_unsized" true
pub fn func_unsized<T: ?Sized>() {}

//@ is "$.index[?(@.name=='func_clone')].inner.function.generics.params[0].kind.type.allow_unsized" false
pub fn func_clone<T: ?Sized + Clone>() {}

//@ is "$.index[?(@.name=='func_debug')].inner.function.generics.params[0].kind.type.allow_unsized" true
pub fn func_debug<T: ?Sized + Debug>() {}

// Generic structs
//@ is "$.index[?(@.name=='StructCustom')].inner.struct.generics.params[0].kind.type.allow_unsized" false
pub struct StructCustom<T: ?Sized + CustomSized>(PhantomData<T>);

//@ is "$.index[?(@.name=='StructUnsized')].inner.struct.generics.params[0].kind.type.allow_unsized" true
pub struct StructUnsized<T: ?Sized>(PhantomData<T>);

//@ is "$.index[?(@.name=='StructClone')].inner.struct.generics.params[0].kind.type.allow_unsized" false
pub struct StructClone<T: ?Sized + Clone>(PhantomData<T>);

//@ is "$.index[?(@.name=='StructDebug')].inner.struct.generics.params[0].kind.type.allow_unsized" true
pub struct StructDebug<T: ?Sized + Debug>(PhantomData<T>);

// Struct with `?Sized` bound, and impl blocks that add additional bounds
//@ is "$.index[?(@.name=='Wrapper')].inner.struct.generics.params[0].kind.type.allow_unsized" true
pub struct Wrapper<T: ?Sized>(PhantomData<T>);

//@ is "$.index[?(@.docs=='impl custom')].inner.impl.generics.params[0].kind.type.allow_unsized" false
/// impl custom
impl<T: ?Sized + CustomSized> Wrapper<T> {
    pub fn impl_custom() {}
}

//@ is "$.index[?(@.docs=='impl clone')].inner.impl.generics.params[0].kind.type.allow_unsized" false
/// impl clone
impl<T: ?Sized + Clone> Wrapper<T> {
    pub fn impl_clone() {}
}

//@ is "$.index[?(@.docs=='impl debug')].inner.impl.generics.params[0].kind.type.allow_unsized" true
/// impl debug
impl<T: ?Sized + Debug> Wrapper<T> {
    pub fn impl_debug() {}
}

impl<T: ?Sized> Wrapper<T> {
    //@ is "$.index[?(@.name=='assoc_custom')].inner.function.generics.params[0].kind.type.allow_unsized" false
    pub fn assoc_custom<U: ?Sized + CustomSized>(&self) {}

    //@ is "$.index[?(@.name=='assoc_unsized')].inner.function.generics.params[0].kind.type.allow_unsized" true
    pub fn assoc_unsized<U: ?Sized>(&self) {}

    //@ is "$.index[?(@.name=='assoc_clone')].inner.function.generics.params[0].kind.type.allow_unsized" false
    pub fn assoc_clone<U: ?Sized + Clone>(&self) {}

    //@ is "$.index[?(@.name=='assoc_debug')].inner.function.generics.params[0].kind.type.allow_unsized" true
    pub fn assoc_debug<U: ?Sized + Debug>(&self) {}
}

// Traits with generic parameters
//@ is "$.index[?(@.name=='TraitCustom')].inner.trait.generics.params[0].kind.type.allow_unsized" false
pub trait TraitCustom<T: ?Sized + CustomSized> {}

//@ is "$.index[?(@.name=='TraitUnsized')].inner.trait.generics.params[0].kind.type.allow_unsized" true
pub trait TraitUnsized<T: ?Sized> {}

//@ is "$.index[?(@.name=='TraitClone')].inner.trait.generics.params[0].kind.type.allow_unsized" false
pub trait TraitClone<T: ?Sized + Clone> {}

//@ is "$.index[?(@.name=='TraitDebug')].inner.trait.generics.params[0].kind.type.allow_unsized" true
pub trait TraitDebug<T: ?Sized + Debug> {}

pub trait TraitMethods {
    //@ is "$.index[?(@.name=='method_custom')].inner.function.generics.params[0].kind.type.allow_unsized" false
    fn method_custom<T: ?Sized + CustomSized>();

    //@ is "$.index[?(@.name=='method_unsized')].inner.function.generics.params[0].kind.type.allow_unsized" true
    fn method_unsized<T: ?Sized>();

    //@ is "$.index[?(@.name=='method_clone')].inner.function.generics.params[0].kind.type.allow_unsized" false
    fn method_clone<T: ?Sized + Clone>();

    //@ is "$.index[?(@.name=='method_debug')].inner.function.generics.params[0].kind.type.allow_unsized" true
    fn method_debug<T: ?Sized + Debug>();
}

// `where` clauses on trait functions, which only affect `T` for that function
//@ is "$.index[?(@.name=='OuterDebug')].inner.trait.generics.params[0].kind.type.allow_unsized" true
pub trait OuterDebug<T: ?Sized> {
    fn foo()
    where
        T: Debug;
}

//@ is "$.index[?(@.name=='OuterClone')].inner.trait.generics.params[0].kind.type.allow_unsized" true
pub trait OuterClone<T: ?Sized> {
    fn foo()
    where
        T: Clone;
}

// Synthetic generic parameters
//@ is "$.index[?(@.name=='synth_clone')].inner.function.generics.params[0].kind.type.allow_unsized" false
pub fn synth_clone(_: impl Clone + ?Sized) {}

//@ is "$.index[?(@.name=='synth_debug')].inner.function.generics.params[0].kind.type.allow_unsized" true
pub fn synth_debug(_: impl Debug + ?Sized) {}
