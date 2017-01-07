#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]

use std::io;

fn try_macro<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    try!(s.write(b"test"));
    //~^ ERROR handle written amount returned
    let mut buf = [0u8; 4];
    try!(s.read(&mut buf));
    //~^ ERROR handle read amount returned
    Ok(())
}

fn question_mark<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    s.write(b"test")?;
    //~^ ERROR handle written amount returned
    let mut buf = [0u8; 4];
    s.read(&mut buf)?;
    //~^ ERROR handle read amount returned
    Ok(())
}

fn unwrap<T: io::Read + io::Write>(s: &mut T) {
    s.write(b"test").unwrap();
    //~^ ERROR handle written amount returned
    let mut buf = [0u8; 4];
    s.read(&mut buf).unwrap();
    //~^ ERROR handle read amount returned
}

fn main() {
}
