fn module() {
    fn test(&mut self) {
    //~^ ERROR `self` parameter is only allowed in associated functions
    }
    mod Self {}
    //~^ ERROR expected identifier, found keyword `Self`
}

fn trait_() {
    fn test(&mut self) {
    //~^ ERROR `self` parameter is only allowed in associated functions
    }
    trait Self {}
    //~^ ERROR expected identifier, found keyword `Self`
}

fn main() {}
