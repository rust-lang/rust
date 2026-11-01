//@ run-pass

unsafe fn call_unsafe(func: unsafe fn() -> ()) -> () {
    func()
}

pub fn main() {
    unsafe { call_unsafe(|| {}); }
}
