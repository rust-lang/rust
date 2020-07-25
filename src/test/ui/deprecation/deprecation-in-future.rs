// check-pass

#![deny(deprecated_in_future)]

#[deprecated(since = "99.99.99", note = "text")]
pub fn deprecated_future() {}

fn test() {
    deprecated_future(); // ok; deprecated_in_future only applies to rustc_deprecated
    //~^ WARNING use of deprecated function `deprecated_future`: text [deprecated]
}

fn main() {}
