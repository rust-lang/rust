enum sty = ~[int];

fn unpack(_unpack: &fn(v: &sty) -> ~[int]) {}

fn main() {
    let _foo = unpack(|s| {
        // Test that `s` is moved here.
        match *s { sty(v) => v } //~ ERROR moving out of dereference of immutable & pointer
    });
}
