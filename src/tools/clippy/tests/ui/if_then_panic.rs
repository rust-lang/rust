// run-rustfix
#![warn(clippy::if_then_panic)]

fn main() {
    let a = vec![1, 2, 3];
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
        println!("qwq");
        println!("qwq");
        println!("qwq");
    }
    if let Some(b) = c {
        panic!("orz {}", b);
    }
    if a.len() == 3 {
        panic!("qaqaq");
    } else {
        println!("qwq");
    }
    let b = vec![1, 2, 3];
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
