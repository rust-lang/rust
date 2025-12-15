//! regression test for issue #47377, #47380
// ignore-tidy-tab
fn main() {
 	let b = "hello";
 	let _a = b + ", World!";
 	//~^ ERROR E0369
}
