// run-pass
#![allow(dead_code, non_camel_case_types, type_param_on_variant_ctor)]

enum opt<T> { none, some(T) }

pub fn main() {
    let x = opt::none::<isize>;
    match x {
        opt::none::<isize> => { println!("hello world"); }
        opt::some(_) => { }
    }
}
