#![allow(clippy::map_identity)]
#![warn(clippy::lines_filter_map_ok)]

use std::io::{self, BufRead, BufReader};

fn main() -> io::Result<()> {
    // Lint:

    let f = std::fs::File::open("/")?;
    BufReader::new(f).lines().filter_map(Result::ok).for_each(|_| ());
    //~^ lines_filter_map_ok
    let f = std::fs::File::open("/")?;
    BufReader::new(f).lines().flat_map(Result::ok).for_each(|_| ());
    //~^ lines_filter_map_ok
    let f = std::fs::File::open("/")?;
    BufReader::new(f).lines().flatten().for_each(|_| ());
    //~^ lines_filter_map_ok

    io::stdin().lines().filter_map(Result::ok).for_each(|_| ());
    //~^ lines_filter_map_ok
    io::stdin().lines().filter_map(|x| x.ok()).for_each(|_| ());
    //~^ lines_filter_map_ok
    io::stdin().lines().flatten().for_each(|_| ());
    //~^ lines_filter_map_ok

    // Do not lint:

    // not a `Lines` iterator
    io::stdin()
        .lines()
        .map(std::convert::identity)
        .filter_map(|x| x.ok())
        .for_each(|_| ());
    // not a `Result::ok()` extractor
    io::stdin().lines().filter_map(|x| x.err()).for_each(|_| ());
    Ok(())
}

#[clippy::msrv = "1.56"]
fn msrv_check() {
    let _lines = BufReader::new(std::fs::File::open("some-path").unwrap())
        .lines()
        .filter_map(Result::ok);
}
