// Regression test for #74280.

#![feature(type_alias_impl_trait)]

type Test = impl Copy;

#[defines(Test)]
fn test() -> Test {
    let y = || -> Test { () };
    7 //~ ERROR mismatched types
}

fn main() {}
