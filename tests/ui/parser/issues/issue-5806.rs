// normalize-stderr-test: "parser:.*\(" -> "parser: $$ACCESS_DENIED_MSG ("
// normalize-stderr-test: "os error \d+" -> "os error $$ACCESS_DENIED_CODE"

#[path = "../parser"]
mod foo; //~ ERROR couldn't read

fn main() {}
