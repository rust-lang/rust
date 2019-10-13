struct ErrorKind;
struct Error(ErrorKind);

impl From<nope::Thing> for Error { //~ ERROR failed to resolve
    fn from(_: nope::Thing) -> Self { //~ ERROR failed to resolve
        unimplemented!()
    }
}

impl From<ErrorKind> for Error {
    fn from(_: ErrorKind) -> Self {
        unimplemented!()
    }
}

fn main() {}
