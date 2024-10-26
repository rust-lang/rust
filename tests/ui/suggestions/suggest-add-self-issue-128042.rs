struct Thing {
    state: u8,
}

impl Thing {
    fn oof(&mut self) { //~ ERROR expected parameter name, found `*`
        self.state = 1;
        //~^ ERROR expected value, found module `self`
    }
}

fn main() {}
