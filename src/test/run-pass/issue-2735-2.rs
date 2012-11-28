// This test should behave exactly like issue-2735-3
struct defer {
    b: &mut bool,
}

impl defer : Drop {
    fn finalize(&self) {
        *(self.b) = true;
    }
}

fn defer(b: &r/mut bool) -> defer/&r {
    defer {
        b: b
    }
}

fn main() {
    let mut dtor_ran = false;
    let _  = defer(&mut dtor_ran);
    assert(dtor_ran);
}
