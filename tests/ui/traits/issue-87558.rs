struct ErrorKind;
struct Error(ErrorKind);
impl Fn(&isize) for Error {
    //~^ ERROR manual implementations of `Fn` are experimental
    //~| ERROR associated item constraints are not allowed here
    //~| ERROR closure, found `Error`
    //~| ERROR not all trait items implemented, missing: `call`
    fn from() {} //~ ERROR method `from` is not a member of trait `Fn`
}

fn main() {}
