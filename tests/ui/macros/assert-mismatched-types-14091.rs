//! Regression test for https://github.com/rust-lang/rust/issues/14091

fn main(){
    assert!(1,1);
    //~^ ERROR mismatched types
}
