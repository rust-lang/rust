#![crate_name = "interesting"]
#![crate_type = "rlib"]

extern crate opaque;

#[no_mangle]
pub fn function_called_always() {
    opaque::opaque_f1();
}

#[no_mangle]
pub fn function_called_never() {
    opaque::opaque_f2();
}

#[no_mangle]
pub fn call_a_bunch_of_functions(fns: &[fn()]) {
    // Indirect call promotion transforms the below into something like
    //
    // for f in fns {
    //     if f == function_called_always {
    //         function_called_always()
    //     } else {
    //         f();
    //     }
    // }
    //
    // where `function_called_always` actually gets inlined too.

    for f in fns {
        f();
    }
}

pub trait Foo {
    fn foo(&self);
}

impl Foo for u32 {
    #[no_mangle]
    fn foo(&self) {
        opaque::opaque_f2();
    }
}

#[no_mangle]
pub fn call_a_bunch_of_trait_methods(trait_objects: &[&dyn Foo]) {
    // Same as above, just with vtables in between
    for x in trait_objects {
        x.foo();
    }
}
