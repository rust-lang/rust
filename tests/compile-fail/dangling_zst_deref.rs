fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32 as *const ()
    };
    let _x = unsafe { *p }; //~ ERROR dangling pointer was dereferenced
}
