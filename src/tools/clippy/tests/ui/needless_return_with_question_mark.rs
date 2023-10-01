//@aux-build:proc_macros.rs
#![allow(
    clippy::needless_return,
    clippy::no_effect,
    clippy::unit_arg,
    clippy::useless_conversion,
    unused
)]

#[macro_use]
extern crate proc_macros;

fn a() -> u32 {
    return 0;
}

fn b() -> Result<u32, u32> {
    return Err(0);
}

// Do not lint
fn c() -> Option<()> {
    return None?;
}

fn main() -> Result<(), ()> {
    return Err(())?;
    return Ok::<(), ()>(());
    Err(())?;
    Ok::<(), ()>(());
    return Err(().into());
    external! {
        return Err(())?;
    }
    with_span! {
        return Err(())?;
    }
    Err(())
}
