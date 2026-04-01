//@ run-rustfix

fn main() {
    let _x: Box<dyn Fn() -> Result<(), ()>> = || { //~ ERROR mismatched types
        Err(())?;
        Ok(())
    };
}
