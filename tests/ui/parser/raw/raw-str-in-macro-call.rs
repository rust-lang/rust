//@ check-pass

macro_rules! m1 {
    ($tt:tt #) => ()
}

macro_rules! m2 {
    ($tt:tt) => ()
}

fn main() {
    m1!(r#"abc"##);
    m2!(r#"abc"#);
}
