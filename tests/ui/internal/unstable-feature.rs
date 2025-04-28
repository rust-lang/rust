#![feature(impl_stability)] 

#[allow_unstable_feature]  //~ ERROR cannot find attribute `allow_unstable_feature` in this scope
//~^ ERROR `allow_unstable_feature` expects a list of feature names
fn main() {

}