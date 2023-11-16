//@aux-build:proc_macros.rs
#![allow(
    clippy::needless_return,
    clippy::no_effect,
    clippy::unit_arg,
    clippy::useless_conversion,
    clippy::diverging_sub_expression,
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

    // Issue #11765
    // Should not lint
    let Some(s) = Some("") else {
        return Err(())?;
    };

    let Some(s) = Some("") else {
        let Some(s) = Some("") else {
            return Err(())?;
        };

        return Err(())?;
    };

    let Some(_): Option<()> = ({
        return Err(())?;
    }) else {
        panic!();
    };

    Err(())
}
