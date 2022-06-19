// ignore-tidy-linelength
use std::mem::ManuallyDrop;

union U1 { // OK
    a: u8,
}

union U2<T: Copy> { // OK
    a: T,
}

union U22<T> { // OK
    a: ManuallyDrop<T>,
}

union U23<T> { // OK
    a: (ManuallyDrop<T>, i32),
}

union U24<T> { // OK
    a: [ManuallyDrop<T>; 2],
}

union U3 {
    a: String, //~ ERROR unions cannot contain fields that may need dropping
}

union U32 { // field that does not drop but is not `Copy`, either -- this is the real feature gate test!
    a: std::cell::RefCell<i32>, //~ ERROR unions with non-`Copy` fields other than `ManuallyDrop<T>`, references, and tuples of such types are unstable
}

union U4<T> {
    a: T, //~ ERROR unions cannot contain fields that may need dropping
}

union U5 { // Having a drop impl is OK
    a: u8,
}

impl Drop for U5 {
    fn drop(&mut self) {}
}

union U5Nested { // a nested union that drops is NOT OK
    nest: U5, //~ ERROR unions cannot contain fields that may need dropping
}

union U6 { // OK
    s: &'static i32,
    m: &'static mut i32,
}

union U7<T> { // OK
    f: (&'static mut i32, ManuallyDrop<T>, i32),
}

fn main() {}
