// This test should behave exactly like issue-2735-3
struct defer {
    let b: &mut bool;
    drop { *(self.b) = true; }
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
