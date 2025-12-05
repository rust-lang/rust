#![warn(clippy::iter_on_empty_collections)]
#![allow(clippy::iter_next_slice, clippy::redundant_clone)]

fn array() {
    assert_eq!([].into_iter().next(), Option::<i32>::None);
    //~^ iter_on_empty_collections
    assert_eq!([].iter_mut().next(), Option::<&mut i32>::None);
    //~^ iter_on_empty_collections
    assert_eq!([].iter().next(), Option::<&i32>::None);
    //~^ iter_on_empty_collections
    assert_eq!(None.into_iter().next(), Option::<i32>::None);
    //~^ iter_on_empty_collections
    assert_eq!(None.iter_mut().next(), Option::<&mut i32>::None);
    //~^ iter_on_empty_collections
    assert_eq!(None.iter().next(), Option::<&i32>::None);
    //~^ iter_on_empty_collections

    // Don't trigger on non-iter methods
    let _: Option<String> = None.clone();
    let _: [String; 0] = [].clone();

    // Don't trigger on match or if branches
    let _ = match 123 {
        123 => [].iter(),
        _ => ["test"].iter(),
    };

    let _ = if false { ["test"].iter() } else { [].iter() };

    let smth = Some(vec![1, 2, 3]);

    // Don't trigger when the empty collection iter is relied upon for its concrete type
    // But do trigger if it is just an iterator, despite being an argument to a method
    for i in smth.as_ref().map_or([].iter(), |s| s.iter()).chain([].iter()) {
        //~^ iter_on_empty_collections
        println!("{i}");
    }

    // Same as above, but for empty collection iters with extra layers
    for i in smth.as_ref().map_or({ [].iter() }, |s| s.iter()) {
        println!("{y}", y = i + 1);
    }

    // Same as above, but for regular function calls
    for i in Option::map_or(smth.as_ref(), [].iter(), |s| s.iter()) {
        println!("{i}");
    }

    // Same as above, but when there are no predicates that mention the collection iter type.
    let mut iter = [34, 228, 35].iter();
    let _ = std::mem::replace(&mut iter, [].iter());
}

macro_rules! in_macros {
    () => {
        assert_eq!([].into_iter().next(), Option::<i32>::None);
        assert_eq!([].iter_mut().next(), Option::<&mut i32>::None);
        assert_eq!([].iter().next(), Option::<&i32>::None);
        assert_eq!(None.into_iter().next(), Option::<i32>::None);
        assert_eq!(None.iter_mut().next(), Option::<&mut i32>::None);
        assert_eq!(None.iter().next(), Option::<&i32>::None);
    };
}

// Don't trigger on a `None` that isn't std's option
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
        None.iter();
        None.iter_mut();
        None.into_iter();
    }
}

fn main() {
    array();
    custom_option::custom_option();
    in_macros!();
}
