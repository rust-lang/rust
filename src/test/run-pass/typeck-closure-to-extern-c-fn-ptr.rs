fn call_extern_c(func: extern "C" fn()) {
    func()
}

pub fn main() {
    call_extern_c(|| {});
}
