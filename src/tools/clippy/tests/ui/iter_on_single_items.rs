#![warn(clippy::iter_on_single_items)]
#![allow(clippy::iter_next_slice, clippy::redundant_clone)]

fn array() {
    assert_eq!([123].into_iter().next(), Some(123));
    //~^ iter_on_single_items
    assert_eq!([123].iter_mut().next(), Some(&mut 123));
    //~^ iter_on_single_items
    assert_eq!([123].iter().next(), Some(&123));
    //~^ iter_on_single_items
    assert_eq!(Some(123).into_iter().next(), Some(123));
    //~^ iter_on_single_items
    assert_eq!(Some(123).iter_mut().next(), Some(&mut 123));
    //~^ iter_on_single_items
    assert_eq!(Some(123).iter().next(), Some(&123));
    //~^ iter_on_single_items

    // Don't trigger on non-iter methods
    let _: Option<String> = Some("test".to_string()).clone();
    let _: [String; 1] = ["test".to_string()].clone();

    // Don't trigger on match or if branches
    let _ = match 123 {
        123 => [].iter(),
        _ => ["test"].iter(),
    };

    let _ = if false { ["test"].iter() } else { [].iter() };
}

macro_rules! in_macros {
    () => {
        assert_eq!([123].into_iter().next(), Some(123));
        assert_eq!([123].iter_mut().next(), Some(&mut 123));
        assert_eq!([123].iter().next(), Some(&123));
        assert_eq!(Some(123).into_iter().next(), Some(123));
        assert_eq!(Some(123).iter_mut().next(), Some(&mut 123));
        assert_eq!(Some(123).iter().next(), Some(&123));
    };
}

// Don't trigger on a `Some` that isn't std's option
mod custom_option {
    #[allow(unused)]
    enum CustomOption {
        Some(i32),
        None,
    }

    impl CustomOption {
        fn iter(&self) {}
        fn iter_mut(&mut self) {}
        fn into_iter(self) {}
    }
    use CustomOption::*;

    pub fn custom_option() {
        Some(3).iter();
        Some(3).iter_mut();
        Some(3).into_iter();
    }
}

fn main() {
    array();
    custom_option::custom_option();
    in_macros!();
}

mod issue14981 {
    use std::option::IntoIter;
    fn takes_into_iter(_: impl IntoIterator<Item = i32>) {}

    fn let_stmt() {
        macro_rules! x {
            ($e:expr) => {
                let _: IntoIter<i32> = $e;
            };
        }
        x!(Some(5).into_iter());
    }

    fn fn_ptr() {
        fn some_func(_: IntoIter<i32>) -> IntoIter<i32> {
            todo!()
        }
        some_func(Some(5).into_iter());

        const C: fn(IntoIter<i32>) -> IntoIter<i32> = <IntoIter<i32> as IntoIterator>::into_iter;
        C(Some(5).into_iter());
    }
}
