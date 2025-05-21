#![allow(non_local_definitions)]
#![warn(clippy::explicit_into_iter_loop)]

fn main() {
    // Issue #4958
    fn _takes_iterator<T>(iterator: &T)
    where
        for<'a> &'a T: IntoIterator<Item = &'a String>,
    {
        for _ in iterator.into_iter() {}
        //~^ explicit_into_iter_loop
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
    //~^ explicit_into_iter_loop

    let r = &t;
    for _ in r.into_iter() {}
    //~^ explicit_into_iter_loop

    // No suggestion for this.
    // We'd have to suggest `for _ in *rr {}` which is less clear.
    let rr = &&t;
    for _ in rr.into_iter() {}

    let mr = &mut t;
    for _ in mr.into_iter() {}
    //~^ explicit_into_iter_loop

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
    //~^ explicit_into_iter_loop

    let mr = &mut u;
    for _ in mr.into_iter() {}
    //~^ explicit_into_iter_loop

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

fn issue14630() {
    macro_rules! mac {
        (into_iter $e:expr) => {
            $e.into_iter()
        };
    }

    for _ in dbg!([1, 2]).into_iter() {}
    //~^ explicit_into_iter_loop

    for _ in mac!(into_iter [1, 2]) {}
}
