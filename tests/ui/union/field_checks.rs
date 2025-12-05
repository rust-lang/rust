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
    a: String, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

union U32 { // field that does not drop but is not `Copy`, either
    a: std::cell::RefCell<i32>, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

union U4<T> {
    a: T, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

union U5 { // Having a drop impl is OK
    a: u8,
}

impl Drop for U5 {
    fn drop(&mut self) {}
}

union U5Nested { // a nested union that drops is NOT OK
    nest: U5, //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

union U5Nested2 { // for now we don't special-case empty arrays
    nest: [U5; 0], //~ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
}

union U6 { // OK
    s: &'static i32,
    m: &'static mut i32,
}

union U7<T> { // OK
    f: (&'static mut i32, ManuallyDrop<T>, i32),
}

union U8<T> { // OK
    f1: [(&'static mut i32, i32); 8],
    f2: [ManuallyDrop<T>; 2],
}

fn main() {}
