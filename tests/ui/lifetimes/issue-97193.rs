extern "C" {
    fn a(&mut self) {
        //~^ ERROR incorrect function inside `extern` block
        //~| ERROR `self` parameter is only allowed in associated functions
        fn b(buf: &Self) {}
    }
}

fn main() {}
