extern "C" {
    fn bget(&self, index: [usize; Self::DIM]) -> bool {
        //~^ ERROR incorrect function inside `extern` block
        //~| ERROR `self` parameter is only allowed in associated functions
        //~| ERROR use of undeclared type `Self`
        type T<'a> = &'a str;
    }
}

fn main() {}
