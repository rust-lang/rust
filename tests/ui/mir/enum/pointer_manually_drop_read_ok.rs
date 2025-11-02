//@ run-pass
//@ compile-flags: -C debug-assertions

#[allow(dead_code)]
#[repr(u16)]
enum Single {
    A,
}

fn main() {
    let illegal_val: u16 = 0;
    let illegal_val_ptr = &raw const illegal_val;
    let foo: *const std::mem::ManuallyDrop<Single> =
        unsafe { std::mem::transmute(illegal_val_ptr) };

    let val: Single = unsafe { foo.cast::<Single>().read() };
    println!("{}", val as u16);
}
