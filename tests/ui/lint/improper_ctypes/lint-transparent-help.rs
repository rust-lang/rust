#![deny(improper_c_fn_definitions)]
use std::marker::PhantomData;
use std::collections::HashMap;
use std::ffi::c_void;

// [option 1] oops, we forgot repr(C)
struct DictPhantom<'a, A,B:'a>{
    value_info: PhantomData<&'a B>,
    full_dict_info: PhantomData<HashMap<A,B>>,
}

#[repr(C)] // [option 2] oops, we meant repr(transparent)
struct MyTypedRawPointer<'a,T:'a>{
    ptr: *const c_void,
    metadata: DictPhantom<'a,T,T>,
}

extern "C" fn example_use(_e: MyTypedRawPointer<i32>) {}
//~^ ERROR: uses type `MyTypedRawPointer<'_, i32>`

fn main() {}
