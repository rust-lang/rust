struct defer {
    x: &[&str],
}

impl defer : Drop {
    fn finalize(&self) {
        error!("%?", self.x);
    }
}

fn defer(x: &r/[&r/str]) -> defer/&r {
    defer {
        x: x
    }
}

fn main() {
    let _x = defer(~["Goodbye", "world!"]); //~ ERROR illegal borrow
}
