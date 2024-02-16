//@ revisions: full min
struct R;

impl R {
    fn method<const N: u8>(&self) -> u8 { N }
}
fn main() {
    assert_eq!(R.method::<1u16>(), 1);
    //~^ ERROR mismatched types
}
