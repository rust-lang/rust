struct Wrapper<T>(T);

impl Wrapper<Option<i32>> {
    fn inner_mut(&self) -> Option<&mut i32> {
        self.as_mut()
        //~^ ERROR no method named `as_mut` found for reference `&Wrapper<Option<i32>>` in the current scope
        //~| HELP one of the expressions' fields has a method of the same name
        //~| HELP items from traits can only be used if
    }

    fn inner_mut_bad(&self) -> Option<&mut u32> {
        self.as_mut()
        //~^ ERROR no method named `as_mut` found for reference `&Wrapper<Option<i32>>` in the current scope
        //~| HELP items from traits can only be used if
    }
}

fn main() {}
