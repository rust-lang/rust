// Macros can be used for (parts of) the pattern and expression in an if let guard
//@ check-pass

#![feature(if_let_guard)]

macro_rules! m {
    (pattern $i:ident) => { Some($i) };
    (expression $e:expr) => { $e };
}

fn main() {
    match () {
        () if let m!(pattern x) = m!(expression Some(4)) => {}
        () if let [m!(pattern y)] = [Some(8 + m!(expression 4))] => {}
        _ => {}
    }
}
