pub struct S;

impl fmt::Debug for S { //~ ERROR failed to resolve: use of undeclared crate or module `fmt`
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { //~ ERROR failed to resolve: use of undeclared crate or module `fmt`
        //~^ ERROR failed to resolve: use of undeclared crate or module `fmt`
        Ok(())
    }
}

fn main() { }
