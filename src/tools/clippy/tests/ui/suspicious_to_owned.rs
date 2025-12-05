//@no-rustfix: overlapping suggestions
#![warn(clippy::suspicious_to_owned)]
#![warn(clippy::implicit_clone)]
#![allow(clippy::redundant_clone)]
use std::borrow::Cow;
use std::ffi::{CStr, c_char};

fn main() {
    let moo = "Moooo";
    let c_moo = b"Moooo\0";
    let c_moo_ptr = c_moo.as_ptr() as *const c_char;
    let moos = ['M', 'o', 'o'];
    let moos_vec = moos.to_vec();

    // we expect this to be linted
    let cow = Cow::Borrowed(moo);
    let _ = cow.to_owned();
    //~^ suspicious_to_owned

    // we expect no lints for this
    let cow = Cow::Borrowed(moo);
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = Cow::Borrowed(moo);
    let _ = cow.clone();

    // we expect this to be linted
    let cow = Cow::Borrowed(&moos);
    let _ = cow.to_owned();
    //~^ suspicious_to_owned

    // we expect no lints for this
    let cow = Cow::Borrowed(&moos);
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = Cow::Borrowed(&moos);
    let _ = cow.clone();

    // we expect this to be linted
    let cow = Cow::Borrowed(&moos_vec);
    let _ = cow.to_owned();
    //~^ suspicious_to_owned

    // we expect no lints for this
    let cow = Cow::Borrowed(&moos_vec);
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = Cow::Borrowed(&moos_vec);
    let _ = cow.clone();

    // we expect this to be linted
    let cow = unsafe { CStr::from_ptr(c_moo_ptr) }.to_string_lossy();
    let _ = cow.to_owned();
    //~^ suspicious_to_owned

    // we expect no lints for this
    let cow = unsafe { CStr::from_ptr(c_moo_ptr) }.to_string_lossy();
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = unsafe { CStr::from_ptr(c_moo_ptr) }.to_string_lossy();
    let _ = cow.clone();

    // we expect no lints for these
    let _ = moo.to_owned();
    let _ = c_moo.to_owned();
    let _ = moos.to_owned();

    // we expect implicit_clone lints for these
    let _ = String::from(moo).to_owned();
    //~^ implicit_clone

    let _ = moos_vec.to_owned();
    //~^ implicit_clone
}
