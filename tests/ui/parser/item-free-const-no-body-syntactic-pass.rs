// Syntactically, a free `const` item can omit its body.

//@ check-pass

fn main() {}

#[cfg(FALSE)]
const X: u8;
