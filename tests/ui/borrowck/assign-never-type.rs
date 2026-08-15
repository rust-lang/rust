// Regression test for issue 62165

//@ check-pass

#![feature(never_type)]

pub fn main() {
    loop {
        match None {
            None => return,
            Some(val) => val,
        };
    };
}
