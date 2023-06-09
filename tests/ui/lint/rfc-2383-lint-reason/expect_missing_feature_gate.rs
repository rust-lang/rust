// should error due to missing feature gate.

#![warn(unused)]

#[expect(unused)]
//~^ ERROR: the `#[expect]` attribute is an experimental feature [E0658]
fn main() {
    let x = 1;
}
