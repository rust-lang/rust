//@ edition:2021
//@ check-pass

union Union {
    value: u64,
}

fn main() {
    let u = Union { value: 42 };

    let c = || {
       unsafe { u.value }
    };

    c();
}
