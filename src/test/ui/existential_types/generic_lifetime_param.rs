// build-pass (FIXME(62277): could be check-pass?)

#![feature(existential_type)]

fn main() {}

existential type Region<'a>: std::fmt::Debug;

fn region<'b>(a: &'b ()) -> Region<'b> {
    a
}
