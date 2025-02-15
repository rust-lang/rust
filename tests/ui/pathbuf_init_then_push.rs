#![warn(clippy::pathbuf_init_then_push)]

use std::path::PathBuf;

fn main() {
    let mut path_buf = PathBuf::new();
    //~^ pathbuf_init_then_push
    path_buf.push("foo");

    path_buf = PathBuf::from("foo");
    //~^ pathbuf_init_then_push
    path_buf.push("bar");

    let bar = "bar";
    path_buf = PathBuf::from("foo");
    //~^ pathbuf_init_then_push
    path_buf.push(bar);

    let mut path_buf = PathBuf::from("foo").join("bar");
    //~^ pathbuf_init_then_push
    path_buf.push("buz");

    let mut x = PathBuf::new();
    println!("{}", x.display());
    x.push("Duck");

    let mut path_buf = PathBuf::new();
    #[cfg(cats)]
    path_buf.push("foo");
}
