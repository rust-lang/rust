#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait HasSelf {
    fn method(self);
}
trait NoSelf {
    fn method();
}
impl NoSelf for u8 {
    reuse HasSelf::method;
    //~^ ERROR this function takes 1 argument but 0 arguments were supplied
}

fn main() {}
