//@ run-fail
//@ error-pattern:oops
//@ needs-subprocess

fn main() {
    let func = || -> ! {
        panic!("oops");
    };
    func();
}
