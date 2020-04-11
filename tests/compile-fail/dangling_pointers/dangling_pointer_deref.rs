fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32
    };
    let x = unsafe { *p }; //~ ERROR dereferenced after this allocation got freed
    panic!("this should never print: {}", x);
}
