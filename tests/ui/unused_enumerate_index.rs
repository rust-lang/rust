#![allow(unused, clippy::map_identity)]
#![warn(clippy::unused_enumerate_index)]

use std::iter::Enumerate;

fn get_enumerate() -> Enumerate<std::vec::IntoIter<i32>> {
    vec![1].into_iter().enumerate()
}

fn main() {
    let v = [1, 2, 3];
    for (_, x) in v.iter().enumerate() {
        //~^ unused_enumerate_index
        println!("{x}");
    }

    struct Dummy1;
    impl Dummy1 {
        fn enumerate(self) -> Vec<usize> {
            vec![]
        }
    }
    let dummy = Dummy1;
    for x in dummy.enumerate() {
        println!("{x}");
    }

    struct Dummy2;
    impl Dummy2 {
        fn enumerate(self) -> Enumerate<std::vec::IntoIter<usize>> {
            vec![1, 2].into_iter().enumerate()
        }
    }
    let dummy = Dummy2;
    for (_, x) in dummy.enumerate() {
        println!("{x}");
    }

    let mut with_used_iterator = [1, 2, 3].into_iter().enumerate();
    with_used_iterator.next();
    for (_, x) in with_used_iterator {
        println!("{x}");
    }

    struct Dummy3(std::vec::IntoIter<usize>);

    impl Iterator for Dummy3 {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            self.0.next()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0.size_hint()
        }
    }

    let dummy = Dummy3(vec![1, 2, 3].into_iter());
    for (_, x) in dummy.enumerate() {
        //~^ unused_enumerate_index
        println!("{x}");
    }

    let _ = vec![1, 2, 3].into_iter().enumerate().map(|(_, x)| println!("{x}"));
    //~^ unused_enumerate_index

    let p = vec![1, 2, 3].into_iter().enumerate();
    //~^ unused_enumerate_index
    p.map(|(_, x)| println!("{x}"));

    // This shouldn't trigger the lint. `get_enumerate` may come from an external library on which we
    // have no control.
    let p = get_enumerate();
    p.map(|(_, x)| println!("{x}"));

    // This shouldn't trigger the lint. The `enumerate` call is in a different context.
    macro_rules! mac {
        () => {
            [1].iter().enumerate()
        };
    }
    _ = mac!().map(|(_, v)| v);

    macro_rules! mac2 {
        () => {
            [1].iter()
        };
    }
    _ = mac2!().enumerate().map(|(_, _v)| {});
    //~^ unused_enumerate_index

    // This shouldn't trigger the lint because of the `allow`.
    #[allow(clippy::unused_enumerate_index)]
    let v = [1].iter().enumerate();
    v.map(|(_, _x)| {});

    // This should keep the explicit type of `x`.
    let v = [1, 2, 3].iter().copied().enumerate();
    //~^ unused_enumerate_index
    let x = v.map(|(_, x): (usize, i32)| x).sum::<i32>();
    assert_eq!(x, 6);

    // This should keep the explicit type of `x`.
    let v = [1, 2, 3].iter().copied().enumerate();
    //~^ unused_enumerate_index
    let x = v.map(|(_, x): (_, i32)| x).sum::<i32>();
    assert_eq!(x, 6);

    let v = [1, 2, 3].iter().copied().enumerate();
    //~^ unused_enumerate_index
    let x = v.map(|(_, x)| x).sum::<i32>();
    assert_eq!(x, 6);
}
