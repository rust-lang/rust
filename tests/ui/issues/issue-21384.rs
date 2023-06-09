// run-pass

use ::std::ops::RangeFull;

fn test<T : Clone>(arg: T) -> T {
    arg.clone()
}

#[derive(PartialEq, Debug)]
struct Test(isize);

fn main() {
    // Check that ranges implement clone
    assert_eq!(test(1..5), (1..5));
    assert_eq!(test(..5), (..5));
    assert_eq!(test(1..), (1..));
    assert_eq!(test(RangeFull), (RangeFull));

    // Check that ranges can still be used with non-clone limits
    assert_eq!((Test(1)..Test(5)), (Test(1)..Test(5)));
}
