//@ check-pass

#![feature(if_let_guard)]

fn split_last(_: &()) -> Option<(&i32, &i32)> {
    None
}

fn assign_twice() {
    loop {
        match () {
            #[allow(irrefutable_let_patterns)]
            () if let _ = split_last(&()) => {}
            _ => {}
        }
    }
}

fn main() {}
