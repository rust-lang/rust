#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code, similar_names, single_match, toplevel_ref_arg, unused_mut, unused_variables)]
#![deny(blacklisted_name)]

fn test(foo: ()) {}

fn main() {
    let foo = 42;
    let bar = 42;
    let baz = 42;

    let barb = 42;
    let barbaric = 42;

    match (42, Some(1337), Some(0)) {
        (foo, Some(bar), baz @ Some(_)) => (),
        _ => (),
    }
}

fn issue_1647(mut foo: u8) {
    let mut bar = 0;
    if let Some(mut baz) = Some(42) {}
}

fn issue_1647_ref() {
    let ref bar = 0;
    if let Some(ref baz) = Some(42) {}
}

fn issue_1647_ref_mut() {
    let ref mut bar = 0;
    if let Some(ref mut baz) = Some(42) {}
}
