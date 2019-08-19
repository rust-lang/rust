// build-pass (FIXME(62277): could be check-pass?)

use std::borrow::Cow;

#[derive(Clone, Debug)]
struct S<'a> {
    name: Cow<'a, str>
}

#[derive(Clone, Debug)]
struct T<'a> {
    s: Cow<'a, [S<'a>]>
}

fn main() {
    let s1 = [S { name: Cow::Borrowed("Test1") }, S { name: Cow::Borrowed("Test2") }];
    let b1 = T { s: Cow::Borrowed(&s1) };
    let s2 = [S { name: Cow::Borrowed("Test3") }, S { name: Cow::Borrowed("Test4") }];
    let b2 = T { s: Cow::Borrowed(&s2) };

    let mut v = Vec::new();
    v.push(b1);
    v.push(b2);

    println!("{:?}", v);
}
