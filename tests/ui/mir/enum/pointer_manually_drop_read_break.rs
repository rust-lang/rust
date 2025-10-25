//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: trying to construct an enum from an invalid value 0x1

#[allow(dead_code)]
#[repr(u16)]
#[derive(Copy, Clone)]
enum Single {
    A,
}

fn main() {
    let illegal_val: u16 = 1;
    let illegal_val_ptr = &raw const illegal_val;
    let foo: *const std::mem::ManuallyDrop<Single> =
        unsafe { std::mem::transmute(illegal_val_ptr) };

    let val: Single = unsafe { std::mem::ManuallyDrop::into_inner(*foo) };
    println!("{}", val as u16);
}
