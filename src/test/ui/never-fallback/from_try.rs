// check-pass
use std::convert::{TryFrom, Infallible};

struct E;

impl From<Infallible> for E {
    fn from(_: Infallible) -> E {
        E
    }
}

fn foo() -> Result<(), E> {
    u32::try_from(1u32)?;
    Ok(())
}

fn main() {}
