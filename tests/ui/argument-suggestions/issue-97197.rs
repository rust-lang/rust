fn main() {
    g((), ());
    //~^ ERROR function takes 6 arguments but 2 arguments were supplied
}

pub fn g(a1: (), a2: bool, a3: bool, a4: bool, a5: bool, a6: ()) -> () {}
