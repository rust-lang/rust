// run-pass
#![allow(dead_code, non_camel_case_types, type_param_on_variant_ctor)]

fn foo<T>(o: myoption<T>) -> isize {
    let mut x: isize = 5;
    match o {
        myoption::none::<T> => { }
        myoption::some::<T>(_) => { x += 1; }
    }
    match o {
        myoption::<T>::none => { }
        myoption::<T>::some(_t) => { x += 1; }
    }
    return x;
}

enum myoption<T> { none, some(T), }

pub fn main() { println!("{}", 5); }
