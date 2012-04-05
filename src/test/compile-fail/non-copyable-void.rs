fn main() {
    let x : *[int] = ptr::addr_of([1,2,3]);
    let y : *libc::c_void = x as *libc::c_void;
    unsafe {
        let _z = *y;
        //!^ ERROR copying a noncopyable value
    }
}
