macro_rules! enum_number {
    ($name:ident { $($variant:ident = $value:expr, )* }) => {
        enum $name {
            $($variant = $value,)*
        }

        fn foo(value: i32) -> Option<$name> {
            match value {
                $( $value => Some($name::$variant), )* // PatKind::Lit
                $( $value ..= 42 => Some($name::$variant), )* // PatKind::Range
                _ => None
            }
        }
    }
}

enum_number!(Change {
    Pos = 1,
    Neg = -1,
    Arith = 1 + 1, //~ ERROR arbitrary expressions aren't allowed in patterns
                   //~^ ERROR only char and numeric types are allowed in range patterns
});

fn main() {}
