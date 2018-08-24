// This tests reification from safe function to `unsafe fn` pointer

fn do_nothing() -> () {}

unsafe fn call_unsafe(func: unsafe fn() -> ()) -> () {
    func()
}

pub fn main() {
    unsafe { call_unsafe(do_nothing); }
}
