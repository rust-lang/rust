//@ edition: 2021

#[macro_export]
macro_rules! make_if {
    (($($tt:tt)*) { $body:expr } { $else:expr }) => {{
        if $($tt)* {
            $body
        } else {
            $else
        }
    }};
    (let ($expr:expr) { $body:expr } { $else:expr }) => {{
        if let None = $expr {
            $body
        } else {
            $else
        }
    }};
    (let ($expr:expr) let ($expr2:expr) { $body:expr } { $else:expr }) => {{
        if let None = $expr && let None = $expr2 {
            $body
        } else {
            $else
        }
    }};
}
