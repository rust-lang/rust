// More checks that stability attributes are used correctly

#![feature(staged_api)]

#![stable(feature = "stable_test_feature", since = "1.0.0")]

#[stable(feature = "a", feature = "b", since = "1.0.0")] //~ ERROR malformed `stable` attribute input [E0538]
fn f1() { }

#[stable(feature = "a", sinse = "1.0.0")] //~ ERROR unknown meta item 'sinse'
fn f2() { }

#[unstable(feature = "a", issue = "no")]
//~^ ERROR `issue` must be a non-zero numeric string or "none"
fn f3() { }

fn main() { }
