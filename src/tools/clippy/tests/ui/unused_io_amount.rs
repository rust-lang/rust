#![allow(dead_code, clippy::needless_pass_by_ref_mut)]
#![allow(clippy::redundant_pattern_matching)]
#![warn(clippy::unused_io_amount)]

extern crate futures;
use futures::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use std::io::{self, Read};

fn question_mark<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    s.write(b"test")?;
    //~^ unused_io_amount
    let mut buf = [0u8; 4];
    s.read(&mut buf)?;
    //~^ unused_io_amount
    Ok(())
}

fn unwrap<T: io::Read + io::Write>(s: &mut T) {
    s.write(b"test").unwrap();
    //~^ unused_io_amount
    let mut buf = [0u8; 4];
    s.read(&mut buf).unwrap();
    //~^ unused_io_amount
}

fn vectored<T: io::Read + io::Write>(s: &mut T) -> io::Result<()> {
    s.read_vectored(&mut [io::IoSliceMut::new(&mut [])])?;
    //~^ unused_io_amount
    s.write_vectored(&[io::IoSlice::new(&[])])?;
    //~^ unused_io_amount
    Ok(())
}

fn ok(file: &str) -> Option<()> {
    let mut reader = std::fs::File::open(file).ok()?;
    let mut result = [0u8; 0];
    reader.read(&mut result).ok()?;
    //~^ unused_io_amount
    Some(())
}

#[allow(clippy::redundant_closure)]
#[allow(clippy::bind_instead_of_map)]
fn or_else(file: &str) -> io::Result<()> {
    let mut reader = std::fs::File::open(file)?;
    let mut result = [0u8; 0];
    reader.read(&mut result).or_else(|err| Err(err))?;
    //~^ unused_io_amount
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
    //~^ unused_io_amount
    Ok(())
}

fn combine_or(file: &str) -> Result<(), Error> {
    let mut reader = std::fs::File::open(file).unwrap();
    let mut result = [0u8; 0];
    reader
        //~^ unused_io_amount
        .read(&mut result)
        .or(Err(Error::Kind))
        .or(Err(Error::Kind))
        .expect("error");
    Ok(())
}

fn is_ok_err<T: io::Read + io::Write>(s: &mut T) {
    s.write(b"ok").is_ok();
    //~^ unused_io_amount
    s.write(b"err").is_err();
    //~^ unused_io_amount
    let mut buf = [0u8; 0];
    s.read(&mut buf).is_ok();
    //~^ unused_io_amount
    s.read(&mut buf).is_err();
    //~^ unused_io_amount
}

async fn bad_async_write<W: AsyncWrite + Unpin>(w: &mut W) {
    w.write(b"hello world").await.unwrap();
    //~^ unused_io_amount
}

async fn bad_async_read<R: AsyncRead + Unpin>(r: &mut R) {
    let mut buf = [0u8; 0];
    r.read(&mut buf[..]).await.unwrap();
    //~^ unused_io_amount
}

async fn io_not_ignored_async_write<W: AsyncWrite + Unpin>(mut w: W) {
    // Here we're forgetting to await the future, so we should get a
    // warning about _that_ (or we would, if it were enabled), but we
    // won't get one about ignoring the return value.
    w.write(b"hello world");
    //~^ unused_io_amount
}

fn bad_async_write_closure<W: AsyncWrite + Unpin + 'static>(w: W) -> impl futures::Future<Output = io::Result<()>> {
    let mut w = w;
    async move {
        w.write(b"hello world").await?;
        //~^ unused_io_amount
        Ok(())
    }
}

async fn async_read_nested_or<R: AsyncRead + Unpin>(r: &mut R, do_it: bool) -> Result<[u8; 1], Error> {
    let mut buf = [0u8; 1];
    if do_it {
        r.read(&mut buf[..]).await.or(Err(Error::Kind))?;
        //~^ unused_io_amount
    }
    Ok(buf)
}

use tokio::io::{AsyncRead as TokioAsyncRead, AsyncReadExt as _, AsyncWrite as TokioAsyncWrite, AsyncWriteExt as _};

async fn bad_async_write_tokio<W: TokioAsyncWrite + Unpin>(w: &mut W) {
    w.write(b"hello world").await.unwrap();
    //~^ unused_io_amount
}

async fn bad_async_read_tokio<R: TokioAsyncRead + Unpin>(r: &mut R) {
    let mut buf = [0u8; 0];
    r.read(&mut buf[..]).await.unwrap();
    //~^ unused_io_amount
}

async fn undetected_bad_async_write<W: AsyncWrite + Unpin>(w: &mut W) {
    // It would be good to detect this case some day, but the current lint
    // doesn't handle it. (The documentation says that this lint "detects
    // only common patterns".)
    let future = w.write(b"Hello world");
    future.await.unwrap();
}

fn match_okay_underscore<T: io::Read + io::Write>(s: &mut T) {
    match s.write(b"test") {
        //~^ unused_io_amount
        Ok(_) => todo!(),
        Err(_) => todo!(),
    };

    let mut buf = [0u8; 4];
    match s.read(&mut buf) {
        //~^ unused_io_amount
        Ok(_) => todo!(),
        Err(_) => todo!(),
    }
}

fn match_okay_underscore_read_expr<T: io::Read + io::Write>(s: &mut T) {
    match s.read(&mut [0u8; 4]) {
        //~^ unused_io_amount
        Ok(_) => todo!(),
        Err(_) => todo!(),
    }
}

fn match_okay_underscore_write_expr<T: io::Read + io::Write>(s: &mut T) {
    match s.write(b"test") {
        //~^ unused_io_amount
        Ok(_) => todo!(),
        Err(_) => todo!(),
    }
}

fn returned_value_should_not_lint<T: io::Read + io::Write>(s: &mut T) -> Result<usize, std::io::Error> {
    s.write(b"test")
}

fn if_okay_underscore_read_expr<T: io::Read + io::Write>(s: &mut T) {
    if let Ok(_) = s.read(&mut [0u8; 4]) {
        //~^ unused_io_amount
        todo!()
    }
}

fn if_okay_underscore_write_expr<T: io::Read + io::Write>(s: &mut T) {
    if let Ok(_) = s.write(b"test") {
        //~^ unused_io_amount
        todo!()
    }
}

fn if_okay_dots_write_expr<T: io::Read + io::Write>(s: &mut T) {
    if let Ok(..) = s.write(b"test") {
        //~^ unused_io_amount
        todo!()
    }
}

fn if_okay_underscore_write_expr_true_negative<T: io::Read + io::Write>(s: &mut T) {
    if let Ok(bound) = s.write(b"test") {
        todo!()
    }
}

fn match_okay_underscore_true_neg<T: io::Read + io::Write>(s: &mut T) {
    match s.write(b"test") {
        Ok(bound) => todo!(),
        Err(_) => todo!(),
    };
}

fn true_negative<T: io::Read + io::Write>(s: &mut T) {
    let mut buf = [0u8; 4];
    let read_amount = s.read(&mut buf).unwrap();
}

fn on_return_should_not_raise<T: io::Read + io::Write>(s: &mut T) -> io::Result<usize> {
    /// this is bad code because it goes around the problem of handling the read amount
    /// by returning it, which makes it impossible to know this is a resonpose from the
    /// correct account.
    let mut buf = [0u8; 4];
    s.read(&mut buf)
}

pub fn unwrap_in_block(rdr: &mut dyn std::io::Read) -> std::io::Result<usize> {
    let read = { rdr.read(&mut [0])? };
    Ok(read)
}

pub fn consumed_example(rdr: &mut dyn std::io::Read) {
    match rdr.read(&mut [0]) {
        Ok(0) => println!("EOF"),
        Ok(_) => println!("fully read"),
        Err(_) => println!("fail"),
    };
    match rdr.read(&mut [0]) {
        Ok(0) => println!("EOF"),
        Ok(_) => println!("fully read"),
        Err(_) => println!("fail"),
    }
}

pub fn unreachable_or_panic(rdr: &mut dyn std::io::Read) {
    {
        match rdr.read(&mut [0]) {
            Ok(_) => unreachable!(),
            Err(_) => println!("expected"),
        }
    }

    {
        match rdr.read(&mut [0]) {
            Ok(_) => panic!(),
            Err(_) => println!("expected"),
        }
    }
}

pub fn wildcards(rdr: &mut dyn std::io::Read) {
    {
        match rdr.read(&mut [0]) {
            Ok(1) => todo!(),
            _ => todo!(),
        }
    }
}
fn allow_works<F: std::io::Read>(mut f: F) {
    let mut data = Vec::with_capacity(100);
    #[allow(clippy::unused_io_amount)]
    f.read(&mut data).unwrap();
}

struct Reader {}

impl Read for Reader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        todo!()
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        // We shouldn't recommend using Read::read_exact inside Read::read_exact!
        self.read(buf).unwrap();
        Ok(())
    }
}

fn main() {}
