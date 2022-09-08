pub trait Ice {
    fn f(&self, _: ());
}

impl Ice for () {
    fn f(&self, _: ()) {}
}

fn main() {
    ().f::<()>(());
    //~^ ERROR this associated function takes 0 generic arguments but 1 generic argument was supplied
}
