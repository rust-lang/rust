fn call_extern_c(func: extern "C" fn()) {
    func()
}

unsafe fn call_unsafe_extern_c(func: unsafe extern "C" fn()) {
    func()
}

pub fn main() {
    call_extern_c(|| {}); //~ ERROR
    unsafe {
        call_unsafe_extern_c(|| {}); //~ ERROR
    }
}
