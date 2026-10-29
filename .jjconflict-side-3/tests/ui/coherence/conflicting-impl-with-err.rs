struct ErrorKind;
struct Error(ErrorKind);

impl From<nope::Thing> for Error { //~ ERROR cannot find
    fn from(_: nope::Thing) -> Self { //~ ERROR cannot find
        unimplemented!()
    }
}

impl From<ErrorKind> for Error {
    fn from(_: ErrorKind) -> Self {
        unimplemented!()
    }
}

fn main() {}
