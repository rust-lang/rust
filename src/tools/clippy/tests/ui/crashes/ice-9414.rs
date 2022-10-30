#![warn(clippy::result_large_err)]

trait T {}
fn f(_: &u32) -> Result<(), *const (dyn '_ + T)> {
    Ok(())
}

fn main() {}
