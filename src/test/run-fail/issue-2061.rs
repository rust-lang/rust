// xfail-test
// error-pattern: ran out of stack
struct R {
    b: int,
}

impl R : Drop {
    fn finalize(&self) {
        let _y = R { b: self.b };
    }
}

fn main() {
    let _x = R { b: 0 };
}
