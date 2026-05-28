//@ edition:2018

#[macro_export]
macro_rules! custom_matches {
    ($expression:expr, $( $pattern:pat )|+ $( if $guard: expr )? $(,)?) => {
        match $expression {
            $( $pattern )|+ $( if $guard )? => true,
            _ => false
        }
    }
}
