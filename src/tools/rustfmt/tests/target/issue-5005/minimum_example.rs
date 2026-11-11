#![feature(more_qualified_paths)]
macro_rules! show {
    ($ty:ty, $ex:expr) => {
        match $ex {
            <$ty>::A(_val) => println!("got a"), // formatting should not remove <$ty>::
            <$ty>::B => println!("got b"),
        }
    };
}
