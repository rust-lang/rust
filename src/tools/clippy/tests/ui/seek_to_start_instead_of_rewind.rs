#![allow(unused)]
#![warn(clippy::seek_to_start_instead_of_rewind)]

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

struct StructWithSeekMethod {}

impl StructWithSeekMethod {
    fn seek(&mut self, from: SeekFrom) {}
}

trait MySeekTrait {
    fn seek(&mut self, from: SeekFrom) {}
}

struct StructWithSeekTrait {}
impl MySeekTrait for StructWithSeekTrait {}

// This should NOT trigger clippy warning because
// StructWithSeekMethod does not implement std::io::Seek;
fn seek_to_start_false_method(t: &mut StructWithSeekMethod) {
    t.seek(SeekFrom::Start(0));
}

// This should NOT trigger clippy warning because
// StructWithSeekMethod does not implement std::io::Seek;
fn seek_to_start_method_owned_false(mut t: StructWithSeekMethod) {
    t.seek(SeekFrom::Start(0));
}

// This should NOT trigger clippy warning because
// StructWithSeekMethod does not implement std::io::Seek;
fn seek_to_start_false_trait(t: &mut StructWithSeekTrait) {
    t.seek(SeekFrom::Start(0));
}

// This should NOT trigger clippy warning because
// StructWithSeekMethod does not implement std::io::Seek;
fn seek_to_start_false_trait_owned(mut t: StructWithSeekTrait) {
    t.seek(SeekFrom::Start(0));
}

// This should NOT trigger clippy warning because
// StructWithSeekMethod does not implement std::io::Seek;
fn seek_to_start_false_trait_bound<T: MySeekTrait>(t: &mut T) {
    t.seek(SeekFrom::Start(0));
}

// This should trigger clippy warning
fn seek_to_start<T: Seek>(t: &mut T) {
    t.seek(SeekFrom::Start(0));
    //~^ seek_to_start_instead_of_rewind
}

// This should trigger clippy warning
fn owned_seek_to_start<T: Seek>(mut t: T) {
    t.seek(SeekFrom::Start(0));
    //~^ seek_to_start_instead_of_rewind
}

// This should NOT trigger clippy warning because
// it does not seek to start
fn seek_to_5<T: Seek>(t: &mut T) {
    t.seek(SeekFrom::Start(5));
}

// This should NOT trigger clippy warning because
// it does not seek to start
fn seek_to_end<T: Seek>(t: &mut T) {
    t.seek(SeekFrom::End(0));
}

// This should NOT trigger clippy warning because
// expr is used here
fn seek_to_start_in_let<T: Seek>(t: &mut T) {
    let a = t.seek(SeekFrom::Start(0)).unwrap();
}

fn main() {
    let mut f = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .truncate(true)
        .open("foo.txt")
        .unwrap();

    let mut my_struct_trait = StructWithSeekTrait {};
    seek_to_start_false_trait_bound(&mut my_struct_trait);

    let hello = "Hello!\n";
    write!(f, "{hello}").unwrap();
    seek_to_5(&mut f);
    seek_to_end(&mut f);
    seek_to_start(&mut f);

    let mut buf = String::new();
    f.read_to_string(&mut buf).unwrap();

    assert_eq!(&buf, hello);
}

#[clippy::msrv = "1.54"]
fn msrv_1_54() {
    let mut f = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .truncate(true)
        .open("foo.txt")
        .unwrap();

    let hello = "Hello!\n";
    write!(f, "{hello}").unwrap();

    f.seek(SeekFrom::Start(0));

    let mut buf = String::new();
    f.read_to_string(&mut buf).unwrap();

    assert_eq!(&buf, hello);
}

#[clippy::msrv = "1.55"]
fn msrv_1_55() {
    let mut f = OpenOptions::new()
        .write(true)
        .read(true)
        .create(true)
        .truncate(true)
        .open("foo.txt")
        .unwrap();

    let hello = "Hello!\n";
    write!(f, "{hello}").unwrap();

    f.seek(SeekFrom::Start(0));
    //~^ seek_to_start_instead_of_rewind

    let mut buf = String::new();
    f.read_to_string(&mut buf).unwrap();

    assert_eq!(&buf, hello);
}
