use std::process::{Command, Stdio};

fn main() {
    let process = Command::new("wc")
        .stdout(Stdio::piped())
        .spawn()
        .or_else(|err| { //~ ERROR type annotations needed
            panic!("oh no: {:?}", err);
        }).unwrap();
}
