extern "C" {
    fn bget(&self, index: [usize; Self::DIM]) -> bool {
        //~^ ERROR incorrect function inside `extern` block
        //~| ERROR `self` parameter is only allowed in associated functions
        //~| ERROR cannot find item `Self`
        type T<'a> = &'a str;
    }
}

fn main() {}
