//@ known-bug: #148891
macro_rules! values {
    ($inner:ty) => {
        #[derive(Debug)]
        pub enum TokenKind {
            #[cfg(test)]
            STRING ([u8; $inner]),
        }
    };
}

values!(String);

pub fn main() {}
