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

    #[expect(unused)]
    fn unused2(){}
}

fn main() {
    let _ = OneUnused;
    let _ = TwoUnused;
}
