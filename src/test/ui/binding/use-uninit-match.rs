// run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]


fn foo<T>(o: myoption<T>) -> isize {
    let mut x: isize = 5;
    match o {
        myoption::none::<T> => { }
        myoption::some::<T>(_t) => { x += 1; }
    }
    return x;
}

enum myoption<T> { none, some(T), }

pub fn main() { println!("{}", 5); }
