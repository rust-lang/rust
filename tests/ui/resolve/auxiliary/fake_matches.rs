// Helper for test tests/ui/resolve/const-with-typo-in-pattern-binding-ice-135289.rs

//@ edition: 2018

#[macro_export]
macro_rules! assert_matches {
    ( $e:expr , $($pat:pat)|+ ) => {
        match $e {
            $($pat)|+ => (),
            _ => (),
        }
    };
}
