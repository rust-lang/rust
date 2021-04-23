use std::result;
impl result { //~ ERROR expected type, found module `result`
    fn into_future() -> Err {} //~ ERROR expected type, found variant `Err`
}
fn main() {}
