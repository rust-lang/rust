// xfail-fast
// xfail-test

fn main() {
    loop foo: {
        loop {
            break foo;
        }
    }
}

