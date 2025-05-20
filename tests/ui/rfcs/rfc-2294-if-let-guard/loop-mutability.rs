//@ check-pass
//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024

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
