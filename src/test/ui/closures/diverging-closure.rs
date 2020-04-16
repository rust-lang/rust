// run-fail
// error-pattern:oops

fn main() {
    let func = || -> ! {
        panic!("oops");
    };
    func();
}
