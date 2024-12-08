#![allow(clippy::all)]
#![deny(clippy::manual_ignore_case_cmp)]

use std::ffi::{OsStr, OsString};

fn main() {}

fn variants(a: &str, b: &str) {
    if a.to_ascii_lowercase() == b.to_ascii_lowercase() {
        return;
    }
    if a.to_ascii_uppercase() == b.to_ascii_uppercase() {
        return;
    }
    let r = a.to_ascii_lowercase() == b.to_ascii_lowercase();
    let r = r || a.to_ascii_uppercase() == b.to_ascii_uppercase();
    r && a.to_ascii_lowercase() == b.to_uppercase().to_ascii_lowercase();
    // !=
    if a.to_ascii_lowercase() != b.to_ascii_lowercase() {
        return;
    }
    if a.to_ascii_uppercase() != b.to_ascii_uppercase() {
        return;
    }
    let r = a.to_ascii_lowercase() != b.to_ascii_lowercase();
    let r = r || a.to_ascii_uppercase() != b.to_ascii_uppercase();
    r && a.to_ascii_lowercase() != b.to_uppercase().to_ascii_lowercase();
}

fn unsupported(a: char, b: char) {
    // TODO:: these are rare, and might not be worth supporting
    a.to_ascii_lowercase() == char::to_ascii_lowercase(&b);
    char::to_ascii_lowercase(&a) == b.to_ascii_lowercase();
    char::to_ascii_lowercase(&a) == char::to_ascii_lowercase(&b);
}

fn char(a: char, b: char) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == *&b.to_ascii_lowercase();
    *&a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == 'a';
    'a' == b.to_ascii_lowercase();
}
fn u8(a: u8, b: u8) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == b'a';
    b'a' == b.to_ascii_lowercase();
}
fn ref_str(a: &str, b: &str) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_uppercase().to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    "a" == b.to_ascii_lowercase();
}
fn ref_ref_str(a: &&str, b: &&str) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_uppercase().to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    "a" == b.to_ascii_lowercase();
}
fn string(a: String, b: String) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    "a" == b.to_ascii_lowercase();
    &a.to_ascii_lowercase() == &b.to_ascii_lowercase();
    &&a.to_ascii_lowercase() == &&b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    "a" == b.to_ascii_lowercase();
}
fn ref_string(a: String, b: &String) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    "a" == b.to_ascii_lowercase();

    b.to_ascii_lowercase() == a.to_ascii_lowercase();
    b.to_ascii_lowercase() == "a";
    "a" == a.to_ascii_lowercase();
}
fn string_ref_str(a: String, b: &str) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    "a" == b.to_ascii_lowercase();

    b.to_ascii_lowercase() == a.to_ascii_lowercase();
    b.to_ascii_lowercase() == "a";
    "a" == a.to_ascii_lowercase();
}
fn ref_u8slice(a: &[u8], b: &[u8]) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
}
fn u8vec(a: Vec<u8>, b: Vec<u8>) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
}
fn ref_u8vec(a: Vec<u8>, b: &Vec<u8>) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    b.to_ascii_lowercase() == a.to_ascii_lowercase();
}
fn ref_osstr(a: &OsStr, b: &OsStr) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
}
fn osstring(a: OsString, b: OsString) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
}
fn ref_osstring(a: OsString, b: &OsString) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    b.to_ascii_lowercase() == a.to_ascii_lowercase();
}
