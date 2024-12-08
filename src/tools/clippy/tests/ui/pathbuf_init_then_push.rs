#![warn(clippy::pathbuf_init_then_push)]

use std::path::PathBuf;

fn main() {
    let mut path_buf = PathBuf::new(); //~ ERROR: calls to `push` immediately after creation
    path_buf.push("foo");

    path_buf = PathBuf::from("foo"); //~ ERROR: calls to `push` immediately after creation
    path_buf.push("bar");

    let bar = "bar";
    path_buf = PathBuf::from("foo"); //~ ERROR: calls to `push` immediately after creation
    path_buf.push(bar);

    let mut path_buf = PathBuf::from("foo").join("bar"); //~ ERROR: calls to `push` immediately after creation
    path_buf.push("buz");

    let mut x = PathBuf::new();
    println!("{}", x.display());
    x.push("Duck");

    let mut path_buf = PathBuf::new();
    #[cfg(cats)]
    path_buf.push("foo");
}
