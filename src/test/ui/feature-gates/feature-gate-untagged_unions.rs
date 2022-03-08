// ignore-tidy-linelength

union U1 { // OK
    a: u8,
}

union U2<T: Copy> { // OK
    a: T,
}

union U22<T> { // OK
    a: std::mem::ManuallyDrop<T>,
}

union U3 {
    a: String, //~ ERROR unions cannot contain fields that may need dropping
}

union U32 { // field that does not drop but is not `Copy`, either -- this is the real feature gate test!
    a: std::cell::RefCell<i32>, //~ ERROR unions with non-`Copy` fields other than `ManuallyDrop<T>` are unstable
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

fn main() {}
