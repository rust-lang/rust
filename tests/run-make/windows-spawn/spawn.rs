use std::io::ErrorKind;
use std::process::Command;

fn main() {
    // Make sure it doesn't try to run "hopefullydoesntexist bar.exe".
    assert_eq!(
        Command::new("hopefullydoesntexist").arg("bar").spawn().unwrap_err().kind(),
        ErrorKind::NotFound
    );
}
