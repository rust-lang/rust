// run-fail
// error-pattern:oh, dear
// ignore-emscripten no processes

fn main() -> ! {
    panic!("oh, dear");
}
