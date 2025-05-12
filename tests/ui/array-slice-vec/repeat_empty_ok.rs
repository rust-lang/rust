#![crate_type = "lib"]

pub struct Header<'a> {
    pub value: &'a [u8],
}

pub fn test() {
    let headers = [Header{value: &[]}; 128];
    //~^ ERROR the trait bound
}

pub fn test2() {
    let headers = [Header{value: &[0]}; 128];
    //~^ ERROR the trait bound
}
