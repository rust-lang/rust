// error-pattern: ran out of stack
struct R {
    b: int,
    drop {
        let _y = R { b: self.b };
    }
}

fn main() {
    let _x = R { b: 0 };
}
