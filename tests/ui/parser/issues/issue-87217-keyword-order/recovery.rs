// test for #115714

struct Misplaced;

impl Misplaced {
    unsafe const fn from_u32(val: u32) {}
    //~^ ERROR expected one of `extern` or `fn`
    fn oof(self){}
}

struct Duplicated;

impl Duplicated {
    unsafe unsafe fn from_u32(val: u32) {}
    //~^ ERROR expected one of `extern` or `fn`
    fn oof(self){}
}

fn main() {
    Misplaced.oof();
    Duplicated.oof();
}
