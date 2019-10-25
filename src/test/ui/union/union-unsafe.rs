#![feature(untagged_unions)]

union U1 {
    a: u8
}

union U2 {
    a: String
}

union U3<T> {
    a: T
}

union U4<T: Copy> {
    a: T
}

fn generic_noncopy<T: Default>() {
    let mut u3 = U3 { a: T::default() };
    u3.a = T::default(); //~ ERROR assignment to non-`Copy` union field is unsafe
}

fn generic_copy<T: Copy + Default>() {
    let mut u3 = U3 { a: T::default() };
    u3.a = T::default(); // OK
    let mut u4 = U4 { a: T::default() };
    u4.a = T::default(); // OK
}

fn main() {
    let mut u1 = U1 { a: 10 }; // OK
    let a = u1.a; //~ ERROR access to union field is unsafe
    u1.a = 11; // OK
    let U1 { a } = u1; //~ ERROR access to union field is unsafe
    if let U1 { a: 12 } = u1 {} //~ ERROR access to union field is unsafe
    // let U1 { .. } = u1; // OK

    let mut u2 = U2 { a: String::from("old") }; // OK
    u2.a = String::from("new"); //~ ERROR assignment to non-`Copy` union field is unsafe
    let mut u3 = U3 { a: 0 }; // OK
    u3.a = 1; // OK
    let mut u3 = U3 { a: String::from("old") }; // OK
    u3.a = String::from("new"); //~ ERROR assignment to non-`Copy` union field is unsafe
}
