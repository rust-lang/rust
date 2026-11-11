// Syntactically, a free `const` item can omit its body.

//@ check-pass

fn main() {}

#[cfg(false)]
const X: u8;
