impl Error for str::Utf8Error {
    //~^ ERROR cannot find trait `Error` in this scope
    //~| ERROR ambiguous associated type
}

fn main() {}
