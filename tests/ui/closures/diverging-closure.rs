//@ run-fail
//@ check-run-results:oops
//@ ignore-emscripten no processes

fn main() {
    let func = || -> ! {
        panic!("oops");
    };
    func();
}
