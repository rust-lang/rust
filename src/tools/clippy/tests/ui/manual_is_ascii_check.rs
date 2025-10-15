#![allow(unused, dead_code)]
#![warn(clippy::manual_is_ascii_check)]

fn main() {
    assert!(matches!('x', 'a'..='z'));
    //~^ manual_is_ascii_check
    assert!(matches!('X', 'A'..='Z'));
    //~^ manual_is_ascii_check
    assert!(matches!(b'x', b'a'..=b'z'));
    //~^ manual_is_ascii_check
    assert!(matches!(b'X', b'A'..=b'Z'));
    //~^ manual_is_ascii_check

    let num = '2';
    assert!(matches!(num, '0'..='9'));
    //~^ manual_is_ascii_check
    assert!(matches!(b'1', b'0'..=b'9'));
    //~^ manual_is_ascii_check
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
    //~^ manual_is_ascii_check

    assert!(matches!('x', 'A'..='Z' | 'a'..='z' | '_'));

    (b'0'..=b'9').contains(&b'0');
    //~^ manual_is_ascii_check
    (b'a'..=b'z').contains(&b'a');
    //~^ manual_is_ascii_check
    (b'A'..=b'Z').contains(&b'A');
    //~^ manual_is_ascii_check

    ('0'..='9').contains(&'0');
    //~^ manual_is_ascii_check
    ('a'..='z').contains(&'a');
    //~^ manual_is_ascii_check
    ('A'..='Z').contains(&'A');
    //~^ manual_is_ascii_check

    let cool_letter = &'g';
    ('0'..='9').contains(cool_letter);
    //~^ manual_is_ascii_check
    ('a'..='z').contains(cool_letter);
    //~^ manual_is_ascii_check
    ('A'..='Z').contains(cool_letter);
    //~^ manual_is_ascii_check
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
    //~^ manual_is_ascii_check
    assert!(matches!('X', 'A'..='Z'));
    //~^ manual_is_ascii_check
    assert!(matches!('x', 'A'..='Z' | 'a'..='z'));
    //~^ manual_is_ascii_check
    assert!(matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F'));
    //~^ manual_is_ascii_check
}

#[clippy::msrv = "1.46"]
fn msrv_1_46() {
    const FOO: bool = matches!('x', '0'..='9');
    const BAR: bool = matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F');
}

#[clippy::msrv = "1.47"]
fn msrv_1_47() {
    const FOO: bool = matches!('x', '0'..='9');
    //~^ manual_is_ascii_check
    const BAR: bool = matches!('x', '0'..='9' | 'a'..='f' | 'A'..='F');
    //~^ manual_is_ascii_check
}

#[allow(clippy::deref_addrof, clippy::needless_borrow)]
fn with_refs() {
    let cool_letter = &&'g';
    ('0'..='9').contains(&&cool_letter);
    //~^ manual_is_ascii_check
    ('a'..='z').contains(*cool_letter);
    //~^ manual_is_ascii_check
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
    //~^ manual_is_ascii_check
    take_while(|c| (b'A'..=b'Z').contains(&c));
    //~^ manual_is_ascii_check
    take_while(|c: char| ('A'..='Z').contains(&c));
    //~^ manual_is_ascii_check
    take_while(|c| matches!(c, 'A'..='Z'));
    //~^ manual_is_ascii_check
}

fn adds_type_reference() {
    let digits: Vec<&char> = ['1', 'A'].iter().take_while(|c| ('0'..='9').contains(c)).collect();
    //~^ manual_is_ascii_check
    let digits: Vec<&mut char> = ['1', 'A'].iter_mut().take_while(|c| ('0'..='9').contains(c)).collect();
    //~^ manual_is_ascii_check
    let digits: Vec<&mut char> = ['1', 'A'].iter_mut().take_while(|c| matches!(c, '0'..='9')).collect();
    //~^ manual_is_ascii_check
}
