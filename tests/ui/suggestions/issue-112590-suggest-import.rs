pub struct S;

impl fmt::Debug for S { //~ ERROR failed to resolve: use of unresolved module or unlinked crate `fmt`
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result { //~ ERROR failed to resolve: use of unresolved module or unlinked crate `fmt`
        //~^ ERROR failed to resolve: use of unresolved module or unlinked crate `fmt`
        Ok(())
    }
}

fn main() { }
