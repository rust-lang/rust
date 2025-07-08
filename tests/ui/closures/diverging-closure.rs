//@ run-fail
//@ check-run-results
//@ needs-subprocess

fn main() {
    let func = || -> ! {
        panic!("oops");
    };
    func();
}
