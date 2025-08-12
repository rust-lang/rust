pub struct S;

impl fmt::Debug for S { //~ ERROR: cannot find `fmt`
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { //~ ERROR: cannot find `fmt`
        //~^ ERROR cannot find `fmt`
        Ok(())
    }
}

fn main() { }
