//@ check-pass

trait Trait<T> {}

fn a(_: impl Trait<
    [(); {
        struct D {
            #[rustfmt::skip]
            bar: (),
        }
        0
    }],
>) {
}

fn main() {}
