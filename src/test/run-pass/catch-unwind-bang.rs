// ignore-wasm32-bare compiled with panic=abort by default

fn worker() -> ! {
    panic!()
}

fn main() {
    std::panic::catch_unwind(worker).unwrap_err();
}
