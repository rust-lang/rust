// The various #[inline(never)] annotations and std::hint::black_box calls are
// an attempt to make unwinding as non-flaky as possible on i686-pc-windows-msvc.

#[inline(never)]
fn generate_backtrace(x: &u32) {
    std::hint::black_box(x);
    let bt = std::backtrace::Backtrace::force_capture();
    println!("{}", bt);
    std::hint::black_box(x);
}

#[inline(never)]
fn fn_in_backtrace(x: &u32) {
    std::hint::black_box(x);
    generate_backtrace(x);
    std::hint::black_box(x);
}

fn main() {
    let x = &41;
    std::hint::black_box(x);
    fn_in_backtrace(x);
    std::hint::black_box(x);
}
