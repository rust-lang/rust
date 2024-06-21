impl Error for str::Utf8Error {
    //~^ ERROR cannot find trait `Error`
    //~| ERROR ambiguous associated type
    fn description(&self)  {}
}

fn main() {}
