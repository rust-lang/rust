//@run-rustfix
#![warn(clippy::explicit_into_iter_loop)]

fn main() {
    // Issue #4958
    fn _takes_iterator<T>(iterator: &T)
    where
        for<'a> &'a T: IntoIterator<Item = &'a String>,
    {
        for _ in iterator.into_iter() {}
    }

    struct T;
    impl IntoIterator for &T {
        type Item = ();
        type IntoIter = std::vec::IntoIter<Self::Item>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }

    let mut t = T;
    for _ in t.into_iter() {}

    let r = &t;
    for _ in r.into_iter() {}

    // No suggestion for this.
    // We'd have to suggest `for _ in *rr {}` which is less clear.
    let rr = &&t;
    for _ in rr.into_iter() {}

    let mr = &mut t;
    for _ in mr.into_iter() {}

    struct U;
    impl IntoIterator for &mut U {
        type Item = ();
        type IntoIter = std::vec::IntoIter<Self::Item>;
        fn into_iter(self) -> Self::IntoIter {
            unimplemented!()
        }
    }

    let mut u = U;
    for _ in u.into_iter() {}

    let mr = &mut u;
    for _ in mr.into_iter() {}

    // Issue #6900
    struct S;
    impl S {
        #[allow(clippy::should_implement_trait)]
        pub fn into_iter<T>(self) -> I<T> {
            unimplemented!()
        }
    }

    struct I<T>(T);
    impl<T> Iterator for I<T> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            unimplemented!()
        }
    }

    for _ in S.into_iter::<u32>() {}
}
