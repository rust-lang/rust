//Missing paren in diagnostic msg: https://github.com/rust-lang/rust/issues/131977
//@check-pass

static mut TEST: usize = 0;

fn main() {
    let _ = unsafe { (&TEST) as *const usize };
    //~^WARN creating a shared reference to mutable static

    let _ = unsafe { (&mut TEST) as *const usize };
    //~^WARN creating a mutable reference to mutable static
}
