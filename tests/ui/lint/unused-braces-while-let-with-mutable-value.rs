//@ check-pass

#![deny(unused_braces)]

fn main() {
    let mut a = Some(3);
    // Shouldn't warn below `a`.
    while let Some(ref mut v) = {a} {
        a.as_mut().map(|a| std::mem::swap(a, v));
        break;
    }
}
