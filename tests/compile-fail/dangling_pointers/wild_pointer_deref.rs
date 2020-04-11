fn main() {
    let p = 44 as *const i32;
    let x = unsafe { *p }; //~ ERROR invalid use of 44 as a pointer
    panic!("this should never print: {}", x);
}
