fn main() {
    let p = {
        let b = Box::new(42);
        &*b as *const i32
    };
    let x = unsafe { *p }; //~ ERROR constant evaluation error
    //~^ NOTE dangling pointer was dereferenced
    panic!("this should never print: {}", x);
}
