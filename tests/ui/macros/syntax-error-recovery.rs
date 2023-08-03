macro_rules! values {
    ($($token:ident($value:literal) $(as $inner:ty)? => $attr:meta,)*) => {
        #[derive(Debug)]
        pub enum TokenKind {
            $(
                #[$attr]
                $token $($inner)? = $value,
            )*
        }
    };
}
//~^^^^^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found invisible open delimiter
//~| ERROR macro expansion ignores invisible open delimiter and any tokens following

values!(STRING(1) as (String) => cfg(test),);
//~^ ERROR expected one of `!` or `::`, found `<eof>`

fn main() {}
