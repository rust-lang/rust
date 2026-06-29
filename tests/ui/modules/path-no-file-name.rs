//@ normalize-stderr: "\.`:.*\(" -> ".`: $$ACCESS_DENIED_MSG ("
//@ normalize-stderr: "os error \d+" -> "os error $$ACCESS_DENIED_CODE"

#[path = "."]
mod m; //~ ERROR `$DIR/.` is a directory
fn main() {}
