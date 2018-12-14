// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::cmp_owned)]
#[allow(clippy::unnecessary_operation)]
fn main() {
    fn with_to_string(x: &str) {
        x != "foo".to_string();

        "foo".to_string() != x;
    }

    let x = "oh";

    with_to_string(x);

    x != "foo".to_owned();

    x != String::from("foo");

    42.to_string() == "42";

    Foo.to_owned() == Foo;

    "abc".chars().filter(|c| c.to_owned() != 'X');

    "abc".chars().filter(|c| *c != 'X');

    let x = &Baz;
    let y = &Baz;

    y.to_owned() == *x;

    let x = &&Baz;
    let y = &Baz;

    y.to_owned() == **x;
}

struct Foo;

impl PartialEq for Foo {
    fn eq(&self, other: &Self) -> bool {
        self.to_owned() == *other
    }
}

impl ToOwned for Foo {
    type Owned = Bar;
    fn to_owned(&self) -> Bar {
        Bar
    }
}

#[derive(PartialEq)]
struct Bar;

impl PartialEq<Foo> for Bar {
    fn eq(&self, _: &Foo) -> bool {
        true
    }
}

impl std::borrow::Borrow<Foo> for Bar {
    fn borrow(&self) -> &Foo {
        static FOO: Foo = Foo;
        &FOO
    }
}

#[derive(PartialEq)]
struct Baz;

impl ToOwned for Baz {
    type Owned = Baz;
    fn to_owned(&self) -> Baz {
        Baz
    }
}
