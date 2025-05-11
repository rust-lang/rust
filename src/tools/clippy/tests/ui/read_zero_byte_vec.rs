#![warn(clippy::read_zero_byte_vec)]
#![allow(
    clippy::unused_io_amount,
    clippy::needless_pass_by_ref_mut,
    clippy::slow_vector_initialization
)]
use std::fs::File;
use std::io;
use std::io::prelude::*;
//@no-rustfix
//@require-annotations-for-level: WARN
extern crate futures;
use futures::io::{AsyncRead, AsyncReadExt};
use tokio::io::{AsyncRead as TokioAsyncRead, AsyncReadExt as _, AsyncWrite as TokioAsyncWrite, AsyncWriteExt as _};

fn test() -> io::Result<()> {
    let cap = 1000;
    let mut f = File::open("foo.txt").unwrap();

    // should lint
    let mut data = Vec::with_capacity(20);
    f.read_exact(&mut data).unwrap();
    //~^ ERROR: reading zero byte data to `Vec`
    //~| NOTE: `-D clippy::read-zero-byte-vec` implied by `-D warnings`

    // should lint
    let mut data2 = Vec::with_capacity(cap);
    f.read_exact(&mut data2)?;
    //~^ ERROR: reading zero byte data to `Vec`

    // should lint
    let mut data3 = Vec::new();
    f.read_exact(&mut data3)?;
    //~^ ERROR: reading zero byte data to `Vec`

    // should lint
    let mut data4 = vec![];
    let _ = f.read(&mut data4)?;
    //~^ ERROR: reading zero byte data to `Vec`

    // should lint
    let _ = {
        let mut data5 = Vec::new();
        f.read(&mut data5)
        //~^ ERROR: reading zero byte data to `Vec`
    };

    // should lint
    let _ = {
        let mut data6: Vec<u8> = Default::default();
        f.read(&mut data6)
        //~^ ERROR: reading zero byte data to `Vec`
    };

    // should not lint
    let mut buf = [0u8; 100];
    f.read(&mut buf)?;

    // should not lint
    let mut data8 = Vec::new();
    data8.resize(100, 0);
    f.read_exact(&mut data8)?;

    // should not lint
    let mut data9 = vec![1, 2, 3];
    f.read_exact(&mut data9)?;

    Ok(())
}

fn test_nested() -> io::Result<()> {
    let cap = 1000;
    let mut f = File::open("foo.txt").unwrap();

    // Issue #9274
    // Should not lint
    let mut v = Vec::new();
    {
        v.resize(10, 0);
        f.read(&mut v)?;
    }

    let mut v = Vec::new();
    {
        f.read(&mut v)?;
        //~^ ERROR: reading zero byte data to `Vec`
    }

    Ok(())
}

async fn test_futures<R: AsyncRead + Unpin>(r: &mut R) {
    // should lint
    let mut data = Vec::new();
    r.read(&mut data).await.unwrap();
    //~^ ERROR: reading zero byte data to `Vec`

    // should lint
    let mut data2 = Vec::new();
    r.read_exact(&mut data2).await.unwrap();
    //~^ ERROR: reading zero byte data to `Vec`
}

async fn test_tokio<R: TokioAsyncRead + Unpin>(r: &mut R) {
    // should lint
    let mut data = Vec::new();
    r.read(&mut data).await.unwrap();
    //~^ ERROR: reading zero byte data to `Vec`

    // should lint
    let mut data2 = Vec::new();
    r.read_exact(&mut data2).await.unwrap();
    //~^ ERROR: reading zero byte data to `Vec`
}

fn allow_works<F: std::io::Read>(mut f: F) {
    let mut data = Vec::with_capacity(100);
    #[allow(clippy::read_zero_byte_vec)]
    f.read(&mut data).unwrap();
}

fn main() {}
