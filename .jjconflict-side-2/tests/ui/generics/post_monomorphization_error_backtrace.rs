//@ build-fail

fn assert_zst<T>() {
    struct F<T>(T);
    impl<T> F<T> {
        const V: () = assert!(std::mem::size_of::<T>() == 0);
        //~^ NOTE: evaluation of `assert_zst::F::<u32>::V` failed
        //~| NOTE: in this expansion of assert!
        //~| ERROR: assertion failed
        //~| NOTE: evaluation of `assert_zst::F::<i32>::V` failed
        //~| NOTE: in this expansion of assert!
        //~| ERROR: assertion failed
    }
    F::<T>::V;
    //~^NOTE: erroneous constant
    //~|NOTE: erroneous constant
    //~|NOTE: duplicate
}

fn foo<U>() {
    assert_zst::<U>()
    //~^ NOTE: the above error was encountered while instantiating `fn assert_zst::<u32>`
    //~| NOTE: the above error was encountered while instantiating `fn assert_zst::<i32>`
}

fn bar<V>() {
    foo::<V>()
}

fn main() {
    bar::<()>();
    bar::<u32>();
    bar::<u32>();
    bar::<i32>();
}
