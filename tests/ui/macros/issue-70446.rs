//@ check-pass

macro_rules! foo {
    ($(: $p:path)? $(: $l:lifetime)? ) => { bar! {$(: $p)? $(: $l)? } };
}

macro_rules! bar {
    ($(: $p:path)? $(: $l:lifetime)? ) => {};
}

foo! {: 'a }

fn main() {}
