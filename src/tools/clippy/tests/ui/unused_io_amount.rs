#![allow(dead_code)]
#![warn(clippy::unused_io_amount)]

use std::io::{self, Read};

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

fn vectored<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    s.read_vectored(&mut [io::IoSliceMut::new(&mut [])])?;
    s.write_vectored(&[io::IoSlice::new(&[])])?;
    Ok(())
}

fn ok(file: &str) -> Option<()> {
    let mut reader = std::fs::File::open(file).ok()?;
    let mut result = [0u8; 0];
    reader.read(&mut result).ok()?;
    Some(())
}

#[allow(clippy::redundant_closure)]
#[allow(clippy::bind_instead_of_map)]
fn or_else(file: &str) -> io::Result<()> {
    let mut reader = std::fs::File::open(file)?;
    let mut result = [0u8; 0];
    reader.read(&mut result).or_else(|err| Err(err))?;
    Ok(())
}

#[derive(Debug)]
enum Error {
    Kind,
}

fn or(file: &str) -> Result<(), Error> {
    let mut reader = std::fs::File::open(file).unwrap();
    let mut result = [0u8; 0];
    reader.read(&mut result).or(Err(Error::Kind))?;
    Ok(())
}

fn combine_or(file: &str) -> Result<(), Error> {
    let mut reader = std::fs::File::open(file).unwrap();
    let mut result = [0u8; 0];
    reader
        .read(&mut result)
        .or(Err(Error::Kind))
        .or(Err(Error::Kind))
        .expect("error");
    Ok(())
}

fn main() {}
