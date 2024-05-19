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
//~^^^^^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found type `(String)`
//~| ERROR macro expansion ignores token `(String)` and any following

values!(STRING(1) as (String) => cfg(test),);
//~^ ERROR expected one of `!` or `::`, found `<eof>`

fn main() {}
