// Regression test for https://github.com/rust-lang/rust/issues/148891.

macro_rules! values {
    ($inner:ty) => {
        #[derive(Debug)]
        pub enum TokenKind {
            #[cfg(test)]
            STRING([u8; $inner]),
            //~^ ERROR expected expression, found `ty` metavariable
            //~| ERROR macro expansion ignores `)` and any tokens following
        }
    };
}

values!(String);

fn main() {}
