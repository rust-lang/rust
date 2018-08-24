// ignore-cross-compile

use std::env;
use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::path::PathBuf;

fn main() {
    if env::args().len() > 1 {
        child().unwrap()
    } else {
        parent().unwrap()
    }
}

fn parent() -> io::Result<()> {
    let td = PathBuf::from(env::var_os("RUST_TEST_TMPDIR").unwrap());
    let input = td.join("stdio-from-input");
    let output = td.join("stdio-from-output");

    File::create(&input)?.write_all(b"foo\n")?;

    // Set up this chain:
    //     $ me <file | me | me >file
    // ... to duplicate each line 8 times total.

    let mut child1 = Command::new(env::current_exe()?)
        .arg("first")
        .stdin(File::open(&input)?) // tests File::into()
        .stdout(Stdio::piped())
        .spawn()?;

    let mut child3 = Command::new(env::current_exe()?)
        .arg("third")
        .stdin(Stdio::piped())
        .stdout(File::create(&output)?) // tests File::into()
        .spawn()?;

    // Started out of order so we can test both `ChildStdin` and `ChildStdout`.
    let mut child2 = Command::new(env::current_exe()?)
        .arg("second")
        .stdin(child1.stdout.take().unwrap()) // tests ChildStdout::into()
        .stdout(child3.stdin.take().unwrap()) // tests ChildStdin::into()
        .spawn()?;

    assert!(child1.wait()?.success());
    assert!(child2.wait()?.success());
    assert!(child3.wait()?.success());

    let mut data = String::new();
    File::open(&output)?.read_to_string(&mut data)?;
    for line in data.lines() {
        assert_eq!(line, "foo");
    }
    assert_eq!(data.lines().count(), 8);
    Ok(())
}

fn child() -> io::Result<()> {
    // double everything
    let mut input = vec![];
    io::stdin().read_to_end(&mut input)?;
    io::stdout().write_all(&input)?;
    io::stdout().write_all(&input)?;
    Ok(())
}
