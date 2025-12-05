//@compile-flags: -Zmiri-disable-stacked-borrows
fn main() {
    let v: Vec<u8> = Vec::with_capacity(10);
    let undef = unsafe { *v.as_ptr().add(5) }; //~ ERROR: uninitialized
    let x = undef + 1;
    panic!("this should never print: {}", x);
}
