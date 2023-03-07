#![feature(associated_type_defaults)]
#![warn(clippy::linkedlist)]
#![allow(unused, dead_code, clippy::needless_pass_by_value)]

extern crate alloc;
use alloc::collections::linked_list::LinkedList;

const C: LinkedList<i32> = LinkedList::new();
static S: LinkedList<i32> = LinkedList::new();

trait Foo {
    type Baz = LinkedList<u8>;
    fn foo(_: LinkedList<u8>);
    const BAR: Option<LinkedList<u8>>;
}

// Ok, we don’t want to warn for implementations; see issue #605.
impl Foo for LinkedList<u8> {
    fn foo(_: LinkedList<u8>) {}
    const BAR: Option<LinkedList<u8>> = None;
}

pub struct Bar {
    priv_linked_list_field: LinkedList<u8>,
    pub pub_linked_list_field: LinkedList<u8>,
}
impl Bar {
    fn foo(_: LinkedList<u8>) {}
}

// All of these test should be trigger the lint because they are not
// part of the public api
fn test(my_favorite_linked_list: LinkedList<u8>) {}
fn test_ret() -> Option<LinkedList<u8>> {
    None
}
fn test_local_not_linted() {
    let _: LinkedList<u8>;
}

// All of these test should be allowed because they are part of the
// public api and `avoid_breaking_exported_api` is `false` by default.
pub fn pub_test(the_most_awesome_linked_list: LinkedList<u8>) {}
pub fn pub_test_ret() -> Option<LinkedList<u8>> {
    None
}

fn main() {}
