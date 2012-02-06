// Issue #901
#[nolink]
native mod libc {
    fn printf(x: ());
}
fn main() { }