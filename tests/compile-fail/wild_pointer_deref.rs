fn main() {
    let p = 42 as *const i32;
    let x = unsafe { *p }; //~ ERROR: tried to access memory through an invalid pointer
    panic!("this should never print: {}", x);
}
