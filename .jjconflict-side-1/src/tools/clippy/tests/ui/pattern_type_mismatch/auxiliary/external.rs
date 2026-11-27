//! **FAKE** external macro crate.

#[macro_export]
macro_rules! macro_with_match {
    ( $p:pat ) => {
        let something = ();

        match &something {
            $p => true,
            _ => false,
        }
    };
}
