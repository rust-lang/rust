// build-fail
#![feature(extern_types)]
#![feature(unsized_fn_params)]
#![crate_type = "lib"]

extern {
    pub type E;
}

fn test(e: E) {} //~ERROR: does not have a dynamically computable size

pub fn calltest(e: Box<E>) {
    test(*e) //~ERROR: does not have a dynamically computable size
}
