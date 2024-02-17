//@ normalize-stderr-test: "\.:.*\(" -> ".: $$ACCESS_DENIED_MSG ("
//@ normalize-stderr-test: "os error \d+" -> "os error $$ACCESS_DENIED_CODE"

#[path = "."]
mod m; //~ ERROR couldn't read

fn main() {}
