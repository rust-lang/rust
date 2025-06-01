struct ErrorKind;
struct Error(ErrorKind);

impl From<nope::Thing> for Error { //~ ERROR cannot find item
    fn from(_: nope::Thing) -> Self { //~ ERROR cannot find item
        unimplemented!()
    }
}

impl From<ErrorKind> for Error {
    fn from(_: ErrorKind) -> Self {
        unimplemented!()
    }
}

fn main() {}
