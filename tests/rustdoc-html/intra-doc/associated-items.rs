#![deny(rustdoc::broken_intra_doc_links)]

/// [`std::collections::BTreeMap::into_iter`]
/// [`String::from`] is ambiguous as to which `From` impl
/// [Vec::into_iter()] uses a disambiguator
//@ has 'associated_items/fn.foo.html' '//a[@href="{{channel}}/alloc/collections/btree/map/type.BTreeMap.html#method.into_iter"]' 'std::collections::BTreeMap::into_iter'
//@ has 'associated_items/fn.foo.html' '//a[@href="{{channel}}/alloc/string/type.String.html#method.from"]' 'String::from'
//@ has 'associated_items/fn.foo.html' '//a[@href="{{channel}}/alloc/vec/type.Vec.html#method.into_iter"]' 'Vec::into_iter'
pub fn foo() {}

/// Link to [MyStruct], [link from struct][MyStruct::method], [MyStruct::clone], [MyStruct::Input]
//@ has 'associated_items/type.MyStruct.html' '//a[@href="type.MyStruct.html"]' 'MyStruct'
//@ has 'associated_items/type.MyStruct.html' '//a[@href="type.MyStruct.html#method.method"]' 'link from struct'
//@ has 'associated_items/type.MyStruct.html' '//a[@href="type.MyStruct.html#method.clone"]' 'MyStruct::clone'
//@ has 'associated_items/type.MyStruct.html' '//a[@href="type.MyStruct.html#associatedtype.Input"]' 'MyStruct::Input'
pub struct MyStruct {
    foo: (),
}

impl Clone for MyStruct {
    fn clone(&self) -> Self {
        MyStruct
    }
}

pub trait T {
    type Input;
    fn method(i: Self::Input);
}

impl T for MyStruct {
    type Input = usize;

    /// [link from method][MyStruct::method] on method
    //@ has 'associated_items/type.MyStruct.html' '//a[@href="type.MyStruct.html#method.method"]' 'link from method'
    fn method(i: usize) {}
}

/// Ambiguity between which trait to use
pub trait T1 {
    fn ambiguous_method();
}

pub trait T2 {
    fn ambiguous_method();
}

/// Link to [S::ambiguous_method]
// FIXME: there is no way to disambiguate these.
// Since we have `#[deny(intra_doc_failure)]`, we still know it was one or the other.
pub struct S;

impl T1 for S {
    fn ambiguous_method() {}
}

impl T2 for S {
    fn ambiguous_method() {}
}

//@ has associated_items/type.MyEnum.html '//a/@href' 'type.MyEnumAlias.html#variant.MyVariant'
/// Link to [MyEnumAlias::MyVariant]
pub enum MyEnum {
    MyVariant,
}

pub type MyEnumAlias = MyEnum;

fn main() {}
