#![warn(clippy::manual_ignore_case_cmp)]
#![allow(
    clippy::deref_addrof,
    clippy::op_ref,
    clippy::ptr_arg,
    clippy::short_circuit_statement,
    clippy::unnecessary_operation
)]

use std::ffi::{OsStr, OsString};

fn main() {}

fn variants(a: &str, b: &str) {
    if a.to_ascii_lowercase() == b.to_ascii_lowercase() {
        //~^ manual_ignore_case_cmp
        return;
    }
    if a.to_ascii_uppercase() == b.to_ascii_uppercase() {
        //~^ manual_ignore_case_cmp
        return;
    }
    let r = a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    let r = r || a.to_ascii_uppercase() == b.to_ascii_uppercase();
    //~^ manual_ignore_case_cmp
    r && a.to_ascii_lowercase() == b.to_uppercase().to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    // !=
    if a.to_ascii_lowercase() != b.to_ascii_lowercase() {
        //~^ manual_ignore_case_cmp
        return;
    }
    if a.to_ascii_uppercase() != b.to_ascii_uppercase() {
        //~^ manual_ignore_case_cmp
        return;
    }
    let r = a.to_ascii_lowercase() != b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    let r = r || a.to_ascii_uppercase() != b.to_ascii_uppercase();
    //~^ manual_ignore_case_cmp
    r && a.to_ascii_lowercase() != b.to_uppercase().to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}

fn unsupported(a: char, b: char) {
    // TODO:: these are rare, and might not be worth supporting
    a.to_ascii_lowercase() == char::to_ascii_lowercase(&b);
    char::to_ascii_lowercase(&a) == b.to_ascii_lowercase();
    char::to_ascii_lowercase(&a) == char::to_ascii_lowercase(&b);
}

fn char(a: char, b: char) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == *&b.to_ascii_lowercase();
    *&a.to_ascii_lowercase() == b.to_ascii_lowercase();
    a.to_ascii_lowercase() == 'a';
    //~^ manual_ignore_case_cmp
    'a' == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn u8(a: u8, b: u8) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == b'a';
    //~^ manual_ignore_case_cmp
    b'a' == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_str(a: &str, b: &str) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_uppercase().to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_ref_str(a: &&str, b: &&str) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_uppercase().to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn string(a: String, b: String) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    &a.to_ascii_lowercase() == &b.to_ascii_lowercase();
    &&a.to_ascii_lowercase() == &&b.to_ascii_lowercase();
    a.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_string(a: String, b: &String) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp

    b.to_ascii_lowercase() == a.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    b.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == a.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn string_ref_str(a: String, b: &str) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    a.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp

    b.to_ascii_lowercase() == a.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    b.to_ascii_lowercase() == "a";
    //~^ manual_ignore_case_cmp
    "a" == a.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_u8slice(a: &[u8], b: &[u8]) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn u8vec(a: Vec<u8>, b: Vec<u8>) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_u8vec(a: Vec<u8>, b: &Vec<u8>) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    b.to_ascii_lowercase() == a.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_osstr(a: &OsStr, b: &OsStr) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn osstring(a: OsString, b: OsString) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
fn ref_osstring(a: OsString, b: &OsString) {
    a.to_ascii_lowercase() == b.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
    b.to_ascii_lowercase() == a.to_ascii_lowercase();
    //~^ manual_ignore_case_cmp
}
