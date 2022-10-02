// run-pass

// ignore-windows - this is a unix-specific test
// ignore-emscripten no processes
// ignore-sgx no processes
use std::os::unix::process::CommandExt;
use std::process::Command;

fn main() {
    let mut command = Command::new("some-boring-name");

    assert_eq!(format!("{:?}", command), r#""some-boring-name""#);

    command.args(&["1", "2", "3"]);

    assert_eq!(format!("{:?}", command), r#""some-boring-name" "1" "2" "3""#);

    command.arg0("exciting-name");

    assert_eq!(format!("{:?}", command), r#"["some-boring-name"] "exciting-name" "1" "2" "3""#);
}
