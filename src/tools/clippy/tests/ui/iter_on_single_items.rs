// run-rustfix
#![warn(clippy::iter_on_single_items)]
#![allow(clippy::iter_next_slice, clippy::redundant_clone)]

fn array() {
    assert_eq!([123].into_iter().next(), Some(123));
    assert_eq!([123].iter_mut().next(), Some(&mut 123));
    assert_eq!([123].iter().next(), Some(&123));
    assert_eq!(Some(123).into_iter().next(), Some(123));
    assert_eq!(Some(123).iter_mut().next(), Some(&mut 123));
    assert_eq!(Some(123).iter().next(), Some(&123));

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
