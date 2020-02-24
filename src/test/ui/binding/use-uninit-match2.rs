// run-pass
#![allow(dead_code, unused_mut, non_camel_case_types, type_param_on_variant_ctor)]


fn foo<T>(o: myoption<T>) -> isize {
    let mut x: isize;
    match o {
        myoption::none::<T> => { panic!(); }
        myoption::some::<T>(_) => { x = 5; }
    }
    let _ = x;
    match o {
        myoption::<T>::none => { panic!(); }
        myoption::<T>::some(_t) => { x = 5; }
    }
    return x;
}

enum myoption<T> { none, some(T), }

pub fn main() { println!("{}", 5); }
