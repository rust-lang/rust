// check-pass

trait Bar {}

impl Bar for () {}
impl Bar for u32 {}

fn takes_closure_ret<F, R>(f: F)
where
    F: FnOnce() -> R,
    R: Bar,
{
}

fn main() {
    takes_closure_ret(|| ());
    // This would normally fallback to ! without v2 fallback algorithm,
    // and then fail because !: Bar is not satisfied.
    takes_closure_ret(|| panic!());
}
