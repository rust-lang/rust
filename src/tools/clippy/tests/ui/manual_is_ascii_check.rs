#![allow(unused, dead_code)]
#![warn(clippy::manual_is_ascii_check)]

fn main() {
    assert!(matches!('x', 'a'..='z'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!(b'x', b'a'..=b'z'));
    assert!(matches!(b'X', b'A'..=b'Z'));

    let num = '2';
    assert!(matches!(num, '0'..='9'));
    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));

    assert!(matches!('x', 'A'..='Z' | 'a'..='z' | '_'));

    (b'0'..=b'9').contains(&b'0');
    (b'a'..=b'z').contains(&b'a');
    (b'A'..=b'Z').contains(&b'A');

    ('0'..='9').contains(&'0');
    ('a'..='z').contains(&'a');
    ('A'..='Z').contains(&'A');

    let cool_letter = &'g';
    ('0'..='9').contains(cool_letter);
    ('a'..='z').contains(cool_letter);
    ('A'..='Z').contains(cool_letter);
}

#[clippy::msrv = "1.23"]
fn msrv_1_23() {
    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
    assert!(matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F'));
}

#[clippy::msrv = "1.24"]
fn msrv_1_24() {
    assert!(matches!(b'1', b'0'..=b'9'));
    assert!(matches!('X', 'A'..='Z'));
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
    assert!(matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F'));
}

#[clippy::msrv = "1.46"]
fn msrv_1_46() {
    const FOO: bool = matches!('x', '0'..='9');
    const BAR: bool = matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F');
}

#[clippy::msrv = "1.47"]
fn msrv_1_47() {
    const FOO: bool = matches!('x', '0'..='9');
    const BAR: bool = matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F');
}

#[allow(clippy::deref_addrof, clippy::needless_borrow)]
fn with_refs() {
    let cool_letter = &&'g';
    ('0'..='9').contains(&&cool_letter);
    ('a'..='z').contains(*cool_letter);
}

fn generics() {
    fn a<U>(u: &U) -> bool
    where
        char: PartialOrd<U>,
        U: PartialOrd<char> + ?Sized,
    {
        ('A'..='Z').contains(u)
    }

    fn take_while<Item, F>(cond: F)
    where
        Item: Sized,
        F: Fn(Item) -> bool,
    {
    }
    take_while(|c| ('A'..='Z').contains(&c));
    take_while(|c| (b'A'..=b'Z').contains(&c));
    take_while(|c: char| ('A'..='Z').contains(&c));
}
