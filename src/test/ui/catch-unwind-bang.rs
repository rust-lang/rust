// run-pass
// needs-unwind

fn worker() -> ! {
    panic!()
}

fn main() {
    std::panic::catch_unwind(worker).unwrap_err();
}
