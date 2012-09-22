// Issue #901
#[nolink]
extern mod libc {
    #[legacy_exports];
    fn printf(x: ());
}
fn main() { }