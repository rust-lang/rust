macro_rules! values {
    ($($token:ident($value:literal) $(as $inner:ty)? => $attr:meta,)*) => {
        #[derive(Debug)]
        pub enum TokenKind {
            $(
                #[$attr]
                $token $($inner)? = $value,
                //~^ ERROR expected one of `!` or `::`, found `<eof>`
            )*
        }
    };
}
//~^^^^^^ ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found `ty` metavariable
//~| ERROR macro expansion ignores `ty` metavariable and any tokens following

values!(STRING(1) as (String) => cfg(test),);

fn main() {}
