#![feature(abi_vectorcall)]
//@ only-x86_64

//@ is "$.index[?(@.name=='AbiVectorcall')].inner.type_alias.type.function_pointer.header.abi.Other" '"\"vectorcall\""'
pub type AbiVectorcall = extern "vectorcall" fn();

//@ is "$.index[?(@.name=='AbiVectorcallUnwind')].inner.type_alias.type.function_pointer.header.abi.Other" '"\"vectorcall-unwind\""'
pub type AbiVectorcallUnwind = extern "vectorcall-unwind" fn();

//@ has "$.index[?(@.name=='Foo')]"
pub struct Foo;

impl Foo {
    //@ is "$.index[?(@.name=='abi_vectorcall')].inner.function.header.abi.Other" '"\"vectorcall\""'
    pub extern "vectorcall" fn abi_vectorcall() {}

    //@ is "$.index[?(@.name=='abi_vectorcall_unwind')].inner.function.header.abi.Other" '"\"vectorcall-unwind\""'
    pub extern "vectorcall-unwind" fn abi_vectorcall_unwind() {}
}

pub trait Bar {
    //@ is "$.index[?(@.name=='trait_abi_vectorcall')].inner.function.header.abi.Other" '"\"vectorcall\""'
    extern "vectorcall" fn trait_abi_vectorcall() {}

    //@ is "$.index[?(@.name=='trait_abi_vectorcall_unwind')].inner.function.header.abi.Other" '"\"vectorcall-unwind\""'
    extern "vectorcall-unwind" fn trait_abi_vectorcall_unwind() {}
}
