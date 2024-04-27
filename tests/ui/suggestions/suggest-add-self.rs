struct X(i32);

impl X {
    pub(crate) fn f() {
        self.0
        //~^ ERROR expected value, found module `self`
    }

    pub fn g() {
        self.0
        //~^ ERROR expected value, found module `self`
    }
}

fn main() {}
