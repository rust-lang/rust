//@ check-pass

struct Foo<const N: bool, const M: u8>;
struct Bar<const N: u8, const M: u32>;

fn main() {
    let _: Foo<true, _> = Foo::<_, 1>;
    let _: Foo<_, 1> = Foo::<true, _>;
    let _: Bar<1, _> = Bar::<_, 300>;
    let _: Bar<_, 300> = Bar::<1, _>;
}
