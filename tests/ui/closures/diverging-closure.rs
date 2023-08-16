// run-fail
//@error-in-other-file:oops
//@ignore-target-emscripten no processes

fn main() {
    let func = || -> ! {
        panic!("oops");
    };
    func();
}
