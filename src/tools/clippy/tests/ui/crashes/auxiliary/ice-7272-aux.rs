pub fn warn<T>(_: T) {}

macro_rules! define_macro {
    ($d:tt $lower:ident $upper:ident) => {
        #[macro_export]
        macro_rules! $upper {
            ($arg:tt) => {
                $crate::$lower($arg)
            };
        }
    };
}

define_macro! {$ warn  WARNING}
