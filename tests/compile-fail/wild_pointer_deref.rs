fn main() {
    let p = 44 as *const i32;
    let x = unsafe { *p }; //~ ERROR dangling pointer was dereferenced
    panic!("this should never print: {}", x);
}
