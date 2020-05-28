pub trait A {}

macro_rules! __implicit_hasher_test_macro {
    (impl< $($impl_arg:tt),* > for $kind:ty where $($bounds:tt)*) => {
        __implicit_hasher_test_macro!( ($($impl_arg),*) ($kind) ($($bounds)*) );
    };

    (($($impl_arg:tt)*) ($($kind_arg:tt)*) ($($bounds:tt)*)) => {
        impl< $($impl_arg)* > test_macro::A for $($kind_arg)* where $($bounds)* { }
    };
}
