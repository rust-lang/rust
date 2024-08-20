macro_rules! enum_number {
    ($name:ident { $($variant:ident = $value:expr, )* }) => {
        enum $name {
            $($variant = $value,)*
        }

        fn foo(value: i32) -> Option<$name> {
            match value {
                $( $value => Some($name::$variant), )* // PatKind::Lit
                //~^ ERROR expected pattern, found expression `1 + 1`
                $( $value ..= 42 => Some($name::$variant), )* // PatKind::Range
                _ => None
            }
        }
    }
}

enum_number!(Change {
    Pos = 1,
    Neg = -1,
    Arith = 1 + 1,
});

fn main() {}
