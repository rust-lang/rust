// run-rustfix
#![crate_type = "lib"]
#![no_std]
#![warn(clippy::if_then_panic)]

pub fn main() {
    let a = &[1, 2, 3];
    let c = Some(2);
    if !a.is_empty()
        && a.len() == 3
        && c != None
        && !a.is_empty()
        && a.len() == 3
        && !a.is_empty()
        && a.len() == 3
        && !a.is_empty()
        && a.len() == 3
    {
        panic!("qaqaq{:?}", a);
    }
    if !a.is_empty() {
        panic!("qaqaq{:?}", a);
    }
    if !a.is_empty() {
        panic!("qwqwq");
    }
    if a.len() == 3 {
        format_args!("qwq");
        format_args!("qwq");
        format_args!("qwq");
    }
    if let Some(b) = c {
        panic!("orz {}", b);
    }
    if a.len() == 3 {
        panic!("qaqaq");
    } else {
        format_args!("qwq");
    }
    let b = &[1, 2, 3];
    if b.is_empty() {
        panic!("panic1");
    }
    if b.is_empty() && a.is_empty() {
        panic!("panic2");
    }
    if a.is_empty() && !b.is_empty() {
        panic!("panic3");
    }
    if b.is_empty() || a.is_empty() {
        panic!("panic4");
    }
    if a.is_empty() || !b.is_empty() {
        panic!("panic5");
    }
}
