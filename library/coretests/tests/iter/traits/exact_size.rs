use std::iter::{InfiniteIterator, once, repeat, repeat_with};

fn test_exact<I: ExactSizeIterator>(iter: I, len: usize) {
    assert_eq!(iter.len(), len);
}

fn test_infinite<I: InfiniteIterator>(_: I) {}

const N: u8 = 42;

#[test]
fn test_exact_size_instance() {
    // repeat(a) is infinite
    test_infinite(repeat(1));

    // RangeFrom is infinite
    test_infinite(0..);

    // iter.cycle() is infinite
    test_infinite(once(1).cycle());

    // finite.chain(infinite) is infinite
    test_infinite(once(1).chain(repeat(1)));

    // FIXME: Reenable, when we get the symmetrical case working
    // // infinite.chain(finite) is infinite
    // test_infinite(repeat(1).chain(once(2)));

    // infinite.chain(infinite) is infinite
    test_infinite(repeat(2).chain(repeat(1)));

    // infinite.cloned() is infinite
    test_infinite(repeat(&N).cloned());

    // infinite.copied() is infinite
    test_infinite(repeat(&N).copied());

    // infinite.enumerate() is infinite
    test_infinite((0..).enumerate());

    // infinite.filter(p) is infinite
    test_infinite((0..).filter(|a| a % 2 == 0));

    // infinite.filter_map(f) is infinite
    test_infinite((0..).filter_map(|a| Some(a % 2 == 0)));

    // infinite.filter_map(f) is infinite
    test_infinite((0..).filter_map(|a| Some(a % 2 == 0)));

    // infinite.flatten() is infinite
    test_infinite(repeat([1]).flatten());

    // infinite.flat_map(f) is infinite
    test_infinite((4..).flat_map(|_| [1, 2]));

    // infinite.inspect(f) is infinite
    test_infinite((1..).inspect(|_| ()));

    // infinite.map(f) is infinite
    test_infinite(repeat(1).map(|a| a));

    // infinite.peekable() is infinite
    test_infinite((0..).peekable());

    // infinite.skip(n) is infinite
    test_infinite((0..).skip(3));

    // infinite.skip_while(p) is infinite
    test_infinite((0..).skip_while(|n| *n < 10));

    // infinite.step_by(n) is infinite
    test_infinite((0..).step_by(10));

    // repeat_with(f) is infinite
    test_infinite(repeat_with(|| 1));

    // infinite.take(n) is exact
    let iter = repeat(1).take(0);
    test_exact(iter, 0);
    let iter = repeat(1).take(3);
    test_exact(iter, 3);

    // infinite.zip(exact) is exact
    let iter = repeat(1).zip(0..7);
    test_exact(iter, 7);
    let iter = repeat(1).zip(0..10);
    test_exact(iter, 10);

    // exact.zip(infinite) is exact
    let iter = (0..7).zip(repeat(1));
    test_exact(iter, 7);
    let iter = (0..10).zip(repeat(1));
    test_exact(iter, 10);
}
