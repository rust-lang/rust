use std::ffi::{OsStr, OsString};
use std::path::Path;

fn check(p: &dyn AsRef<Path>) {
    let m = std::fs::metadata(&p);
    println!("{:?}", &m);
}

fn main() {
    let s: OsString = ".".into();
    let s: &OsStr = &s;
    check(s);
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
    //~| HELP within `OsStr`, the trait `Sized` is not implemented for `[u8]`
    //~| HELP consider borrowing the value, since `&OsStr` can be coerced into `dyn AsRef<Path>`
}
