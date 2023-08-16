// run-fail
//@error-in-other-file:oh, dear
//@ignore-target-emscripten no processes

fn main() -> ! {
    panic!("oh, dear");
}
