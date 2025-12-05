//@ only-windows
//@ run-pass
//@ run-flags:--parent-process

use std::env;
use std::io::ErrorKind::{self, InvalidInput};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    if env::args().nth(1).as_deref() == Some("--parent-process") {
        parent();
    } else {
        child();
    }
}

fn child() {
    if env::args().len() == 1 {
        panic!("something went wrong :/");
    }
    for arg in env::args().skip(1) {
        print!("{arg}\0");
    }
}

fn parent() {
    let mut bat = PathBuf::from(file!());
    bat.set_file_name("windows-bat-args1.bat");
    let bat1 = String::from(bat.to_str().unwrap());
    bat.set_file_name("windows-bat-args2.bat");
    let bat2 = String::from(bat.to_str().unwrap());
    bat.set_file_name("windows-bat-args3.bat");
    let bat3 = String::from(bat.to_str().unwrap());
    bat.set_file_name("windows-bat-args1.bat .. ");
    let bat4 = String::from(bat.to_str().unwrap());
    let bat = [bat1.as_str(), bat2.as_str(), bat3.as_str(), bat4.as_str()];

    check_args(&bat, &["a", "b"]).unwrap();
    check_args(&bat, &["c is for cat", "d is for dog"]).unwrap();
    check_args(&bat, &["\"", " \""]).unwrap();
    check_args(&bat, &["\\", "\\"]).unwrap();
    check_args(&bat, &[">file.txt"]).unwrap();
    check_args(&bat, &["whoami.exe"]).unwrap();
    check_args(&bat, &["&a.exe"]).unwrap();
    check_args(&bat, &["&echo hello "]).unwrap();
    check_args(&bat, &["&echo hello", "&whoami", ">file.txt"]).unwrap();
    check_args(&bat, &["!TMP!"]).unwrap();
    check_args(&bat, &["key=value"]).unwrap();
    check_args(&bat, &["\"key=value\""]).unwrap();
    check_args(&bat, &["key = value"]).unwrap();
    check_args(&bat, &["key=[\"value\"]"]).unwrap();
    check_args(&bat, &["", "a=b"]).unwrap();
    check_args(&bat, &["key=\"foo bar\""]).unwrap();
    check_args(&bat, &["key=[\"my_value]"]).unwrap();
    check_args(&bat, &["key=[\"my_value\",\"other-value\"]"]).unwrap();
    check_args(&bat, &["key\\=value"]).unwrap();
    check_args(&bat, &["key=\"&whoami\""]).unwrap();
    check_args(&bat, &["key=\"value\"=5"]).unwrap();
    check_args(&bat, &["key=[\">file.txt\"]"]).unwrap();
    assert_eq!(check_args(&bat, &["\n"]), Err(InvalidInput));
    assert_eq!(check_args(&bat, &["\r"]), Err(InvalidInput));
    check_args(&bat, &["%hello"]).unwrap();
    check_args(&bat, &["%PATH%"]).unwrap();
    check_args(&bat, &["%%cd:~,%"]).unwrap();
    check_args(&bat, &["%PATH%PATH%"]).unwrap();
    check_args(&bat, &["\">file.txt"]).unwrap();
    check_args(&bat, &["abc\"&echo hello"]).unwrap();
    check_args(&bat, &["123\">file.txt"]).unwrap();
    check_args(&bat, &["\"&echo hello&whoami.exe"]).unwrap();
    check_args(&bat, &[r#"hello^"world"#, "hello &echo oh no >file.txt"]).unwrap();
}

// Check if the arguments roundtrip through a bat file and back into a Rust process.
// Our Rust process outptuts the arguments as null terminated strings.
#[track_caller]
fn check_args(bats: &[&str], args: &[&str]) -> Result<(), ErrorKind> {
    for bat in bats {
        let output = Command::new(&bat).args(args).output().map_err(|e| e.kind())?;
        assert!(output.status.success());
        let child_args = String::from_utf8(output.stdout).unwrap();
        let mut child_args: Vec<&str> =
            child_args.strip_suffix('\0').unwrap().split('\0').collect();
        // args3.bat can append spurious empty arguments, so trim them here.
        child_args.truncate(
            child_args.iter().rposition(|s| !s.is_empty()).unwrap_or(child_args.len() - 1) + 1,
        );
        assert_eq!(&child_args, &args);
        assert!(!Path::new("file.txt").exists());
    }
    Ok(())
}
