// Regression test for #75883.

pub struct UI {}

impl UI {
    pub fn run() -> Result<_> {
        //~^ ERROR: this enum takes 2 generic arguments but 1 generic argument was supplied
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for return types
        let mut ui = UI {};
        ui.interact();

        unimplemented!();
    }

    pub fn interact(&mut self) -> Result<_> {
        //~^ ERROR: this enum takes 2 generic arguments but 1 generic argument was supplied
        //~| ERROR: the placeholder `_` is not allowed within types on item signatures for return types
        unimplemented!();
    }
}

fn main() {}
