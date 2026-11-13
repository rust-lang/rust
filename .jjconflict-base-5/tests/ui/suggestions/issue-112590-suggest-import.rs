pub struct S;

impl fmt::Debug for S { //~ ERROR: cannot find module or crate `fmt`
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        //~^ ERROR: cannot find module or crate `fmt`
        //~| ERROR: cannot find module or crate `fmt`
        Ok(())
    }
}

fn main() { }
