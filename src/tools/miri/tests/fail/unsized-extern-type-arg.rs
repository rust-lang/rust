#![feature(extern_types)]
#![feature(unsized_fn_params)]

extern {
    pub type E;
}

fn test(_e: E) {}

pub fn calltest(e: Box<E>) {
    test(*e) //~ERROR: does not have a dynamically computable size
}

fn main() {
    let b = Box::new(0u32);
    calltest(unsafe { std::mem::transmute(b)} );
}
