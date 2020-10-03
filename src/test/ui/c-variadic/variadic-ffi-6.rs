#![crate_type="lib"]
#![feature(c_variadic)]

pub unsafe extern "C" fn use_vararg_lifetime(
    x: usize,
    y: ...
) -> &usize { //~ ERROR missing lifetime specifier
    &0
}

pub unsafe extern "C" fn use_normal_arg_lifetime(x: &usize, y: ...) -> &usize { // OK
    x
}

#[repr(C)]
pub struct Foo(usize);


impl Foo {
    #[no_mangle]
    pub unsafe extern "C" fn assoc_fn(_format: *const i8, ap: ...) -> usize { // OK
        ap.arg()
    }

    #[no_mangle]
    pub unsafe extern "C" fn method(&self, _format: *const i8, ap: ...) -> usize { // OK
        ap.arg()
    }
}
