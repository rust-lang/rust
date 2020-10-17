// run-rustfix

#![warn(clippy::search_is_some)]

fn main() {
    let v = vec![3, 2, 1, 0, -1, -2, -3];
    let y = &&42;

    // Check `find().is_some()`, single-line case.
    let _ = v.iter().find(|&x| *x < 0).is_some();
    let _ = (0..1).find(|x| **y == *x).is_some(); // one dereference less
    let _ = (0..1).find(|x| *x == 0).is_some();
    let _ = v.iter().find(|x| **x == 0).is_some();

    // Check `position().is_some()`, single-line case.
    let _ = v.iter().position(|&x| x < 0).is_some();

    // Check `rposition().is_some()`, single-line case.
    let _ = v.iter().rposition(|&x| x < 0).is_some();

    let s1 = String::from("hello world");
    let s2 = String::from("world");
    // caller of `find()` is a `&`static str`
    let _ = "hello world".find("world").is_some();
    let _ = "hello world".find(&s2).is_some();
    let _ = "hello world".find(&s2[2..]).is_some();
    // caller of `find()` is a `String`
    let _ = s1.find("world").is_some();
    let _ = s1.find(&s2).is_some();
    let _ = s1.find(&s2[2..]).is_some();
    // caller of `find()` is slice of `String`
    let _ = s1[2..].find("world").is_some();
    let _ = s1[2..].find(&s2).is_some();
    let _ = s1[2..].find(&s2[2..]).is_some();
}
