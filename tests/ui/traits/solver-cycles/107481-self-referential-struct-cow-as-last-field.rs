// Regression test for #107481

//@ check-pass

use std::{borrow::Cow, collections::HashMap};

#[derive(Clone)]
struct Foo<'a>(Cow<'a, [Self]>);

#[derive(Clone)]
struct Bar<'a>(Cow<'a, HashMap<String, Self>>);

#[derive(Clone)]
struct Baz<'a>(Cow<'a, Vec<Self>>);

#[derive(Clone)]
struct Qux<'a>(Cow<'a, Box<Self>>);

fn main() {}
