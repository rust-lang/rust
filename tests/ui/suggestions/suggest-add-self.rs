struct X(i32);

impl X {
    pub(crate) fn f() {
        self.0
        //~^ ERROR cannot find value `self` in this scope
    }

    pub fn g() {
        self.0
        //~^ ERROR cannot find value `self` in this scope
    }
}

fn main() {}
