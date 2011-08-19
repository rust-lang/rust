// -*- rust -*-
use std;
import std::io;
import std::str;

#[cfg(target_os = "linux")]
#[cfg(target_os = "win32")]
#[test]
fn test_simple() {
    let tmpfile: str = "test/run-pass/lib-io-test-simple.tmp";
    log tmpfile;
    let frood: str = "A hoopy frood who really knows where his towel is.";
    log frood;
    {
        let out: io::writer =
            io::file_writer(tmpfile, [io::create, io::truncate]);
        out.write_str(frood);
    }
    let inp: io::reader = io::file_reader(tmpfile);
    let frood2: str = inp.read_c_str();
    log frood2;
    assert (str::eq(frood, frood2));
}

// FIXME (726)
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn test_simple() { }

