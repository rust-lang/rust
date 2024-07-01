pub struct S;

impl fmt::Debug for S { //~ ERROR cannot find item `fmt`
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { //~ ERROR cannot find item `fmt`
        //~^ ERROR cannot find item `fmt`
        Ok(())
    }
}

fn main() { }
