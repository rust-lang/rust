// run-fail
// error-pattern:oops
// ignore-emscripten no processes

fn main() {
    let func = || -> ! {
        panic!("oops");
    };
    func();
}
