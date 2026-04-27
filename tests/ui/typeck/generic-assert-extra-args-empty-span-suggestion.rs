// Regression test for issue #131762.

#![feature(generic_assert)]

struct FloatWrapper(f64);

fn main() {
    assert!(
        (0.0 / 0.0 >= 0.0)
            == (FloatWrapper(0.0 / 0.0)
                >= FloatWrapper( //~ ERROR binary operation `>=` cannot be applied to type `FloatWrapper`
                //~| ERROR this struct takes 1 argument but 3 arguments were supplied
                //~| ERROR this struct takes 1 argument but 3 arguments were supplied
                    size_of::<u8>,
                    size_of::<u16>,
                    size_of::<usize> as fn() -> usize,
                ))
    );
}
