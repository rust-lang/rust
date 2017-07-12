fn main() {
    let x = Box::new(42);
    {
        let bad_box: Box<i32> = unsafe { std::ptr::read(&x) };
        drop(bad_box);
    }
    drop(x); //~ ERROR dangling pointer was dereferenced
}
