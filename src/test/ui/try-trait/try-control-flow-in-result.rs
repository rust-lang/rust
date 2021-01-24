#![feature(control_flow_enum)]
#![feature(try_trait_v2)]

use std::ops::ControlFlow;

fn returns_control_flow() -> ControlFlow<()> {
    ControlFlow::BREAK
}

fn demo() -> Result<(), ()> {
    returns_control_flow()?; //~ ERROR the `?` operator can only be used in a function that
    Ok(())
}

fn main() {}
