unsafe fn call_unsafe_extern_c(func: unsafe extern "C" fn()) {
    func()
}

pub fn main() {
    unsafe { call_unsafe_extern_c(|| {}); }
}
