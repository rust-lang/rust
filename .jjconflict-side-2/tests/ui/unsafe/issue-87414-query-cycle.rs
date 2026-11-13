// Regression test for #87414.

//@ check-pass

fn bad<T>() -> Box<dyn Iterator<Item = [(); { |x: u32| { x }; 4 }]>> { todo!() }

fn foo() -> [(); { |x: u32| { x }; 4 }] { todo!() }
fn bar() { let _: [(); { |x: u32| { x }; 4 }]; }

// This one should not cause any errors either:
unsafe fn unsf() {}
fn bad2<T>() -> Box<dyn Iterator<Item = [(); { unsafe { || { unsf() } }; 4 }]>> { todo!() }

fn main() {}
