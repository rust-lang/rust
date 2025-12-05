//@ check-pass

macro_rules! foo {
    () => {
        #[allow(unreachable_patterns)]
        {
            123i32
        }
    };
}

fn main() {
    let _ = foo!().abs();
}
