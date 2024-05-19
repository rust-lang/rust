//@ run-pass
macro_rules! enum_number {
    ($name:ident { $($variant:ident = $value:expr, )* }) => {
        enum $name {
            $($variant = $value,)*
        }

        fn foo(value: i32) -> Option<$name> {
            match value {
                $( $value => Some($name::$variant), )*
                _ => None
            }
        }
    }
}

enum_number!(Change {
    Down = -1,
    None = 0,
    Up = 1,
});

fn main() {
    if let Some(Change::Down) = foo(-1) {} else { panic!() }
}
