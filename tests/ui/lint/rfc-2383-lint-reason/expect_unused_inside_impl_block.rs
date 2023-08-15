// check-pass

#![feature(lint_reasons)]
#![warn(unused)]

struct OneUnused;
struct TwoUnused;

impl OneUnused {
    #[expect(unused)]
    fn unused() {}
}

impl TwoUnused {
    #[expect(unused)]
    fn unused1(){}

    // Tests a regression where the compiler erroneously determined that all `#[expect(unused)]`
    // after the first method in the impl block were unfulfilled.
    // issue 114416
    #[expect(unused)]
    fn unused2(){}
}

fn main() {
    let _ = OneUnused;
    let _ = TwoUnused;
}
