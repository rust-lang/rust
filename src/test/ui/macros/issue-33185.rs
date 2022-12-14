// run-pass
#![allow(dead_code)]

#[macro_export]
macro_rules! state {
    ( $( $name:ident : $field:ty )* ) => (
        #[derive(Default)]
        struct State {
            $($name : $field),*
        }
    )
}

state! { x: i64 }

pub fn main() {
}
