//@no-rustfix: overlapping suggestions
#![warn(clippy::suspicious_to_owned)]
#![warn(clippy::implicit_clone)]
#![allow(clippy::redundant_clone)]
use std::borrow::Cow;
use std::ffi::{c_char, CStr};

fn main() {
    let moo = "Moooo";
    let c_moo = b"Moooo\0";
    let c_moo_ptr = c_moo.as_ptr() as *const c_char;
    let moos = ['M', 'o', 'o'];
    let moos_vec = moos.to_vec();

    // we expect this to be linted
    let cow = Cow::Borrowed(moo);
    let _ = cow.to_owned();
    //~^ ERROR: this `to_owned` call clones the Cow<'_, str> itself and does not cause the
    //~| NOTE: `-D clippy::suspicious-to-owned` implied by `-D warnings`
    // we expect no lints for this
    let cow = Cow::Borrowed(moo);
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = Cow::Borrowed(moo);
    let _ = cow.clone();

    // we expect this to be linted
    let cow = Cow::Borrowed(&moos);
    let _ = cow.to_owned();
    //~^ ERROR: this `to_owned` call clones the Cow<'_, [char; 3]> itself and does not cau
    // we expect no lints for this
    let cow = Cow::Borrowed(&moos);
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = Cow::Borrowed(&moos);
    let _ = cow.clone();

    // we expect this to be linted
    let cow = Cow::Borrowed(&moos_vec);
    let _ = cow.to_owned();
    //~^ ERROR: this `to_owned` call clones the Cow<'_, Vec<char>> itself and does not cau
    // we expect no lints for this
    let cow = Cow::Borrowed(&moos_vec);
    let _ = cow.into_owned();
    // we expect no lints for this
    let cow = Cow::Borrowed(&moos_vec);
    let _ = cow.clone();

    // we expect this to be linted
    let cow = unsafe { CStr::from_ptr(c_moo_ptr) }.to_string_lossy();
    let _ = cow.to_owned();
    //~^ ERROR: this `to_owned` call clones the Cow<'_, str> itself and does not cause the
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
    //~^ ERROR: implicitly cloning a `String` by calling `to_owned` on its dereferenced ty
    //~| NOTE: `-D clippy::implicit-clone` implied by `-D warnings`
    let _ = moos_vec.to_owned();
    //~^ ERROR: implicitly cloning a `Vec` by calling `to_owned` on its dereferenced type
}
