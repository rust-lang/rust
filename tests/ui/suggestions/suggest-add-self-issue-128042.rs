struct Thing {
    state: u8,
}

impl Thing {
    fn oof(*mut Self) { //~ ERROR expected parameter name, found `*`
        self.state = 1;
        //~^ ERROR cannot find value `self` in this scope
    }
}

fn main() {}
