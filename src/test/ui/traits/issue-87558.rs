struct ErrorKind;
struct Error(ErrorKind);
impl Fn(&isize) for Error {
    //~^ ERROR manual implementations of `Fn` are experimental
    //~| ERROR associated type bindings are not allowed here
    fn from() {} //~ ERROR method `from` is not a member of trait `Fn`
}

fn main() {}
