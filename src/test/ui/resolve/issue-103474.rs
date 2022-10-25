struct S {}
impl S {
    fn first(&self) {}

    fn second(&self) {
        first()
        //~^ ERROR cannot find function `first` in this scope
    }

    fn third(&self) {
        no_method_err()
        //~^ ERROR cannot find function `no_method_err` in this scope
    }
}

fn main() {}
