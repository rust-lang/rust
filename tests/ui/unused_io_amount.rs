#![allow(dead_code)]
#![warn(clippy::unused_io_amount)]

use std::io;

fn try_macro<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    r#try!(s.write(b"test"));
    let mut buf = [0u8; 4];
    r#try!(s.read(&mut buf));
    Ok(())
}

fn question_mark<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    s.write(b"test")?;
    let mut buf = [0u8; 4];
    s.read(&mut buf)?;
    Ok(())
}

fn unwrap<T: io::Read + io::Write>(s: &mut T) {
    s.write(b"test").unwrap();
    let mut buf = [0u8; 4];
    s.read(&mut buf).unwrap();
}

fn main() {}
