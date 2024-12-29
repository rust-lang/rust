//@ run-fail
//@ check-run-results:oh, dear
//@ ignore-emscripten no processes

fn main() -> ! {
    panic!("oh, dear");
}
