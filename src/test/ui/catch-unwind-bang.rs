// run-pass
// ignore-emscripten compiled with panic=abort by default

fn worker() -> ! {
    panic!()
}

fn main() {
    std::panic::catch_unwind(worker).unwrap_err();
}
