// FIXME: This test is transitional until estrs are gone.
use std;
fn main() {
    let s = #fmt[~"%S", ~"test"];
    assert s == "test";
}