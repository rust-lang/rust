/// Discovered in https://github.com/rust-lang/rust/issues/112602.
/// This caused a cycle error, which made no sense.
/// Removing the `const` part of the `many` function would make the
/// test pass again.
/// The issue was that we were running const qualif checks on
/// `const fn`s, but never using them. During const qualif checks we tend
/// to end up revealing opaque types (the RPIT in `many`'s return type),
/// which can quickly lead to cycles.

//@ check-pass

pub struct Parser<H>(H);

impl<H, T> Parser<H>
where
    H: for<'a> Fn(&'a str) -> T,
{
    pub const fn new(handler: H) -> Parser<H> {
        Parser(handler)
    }

    pub const fn many<'s>(&'s self) -> Parser<impl for<'a> Fn(&'a str) -> Vec<T> + 's> {
        Parser::new(|_| unimplemented!())
    }
}

fn main() {
    println!("Hello, world!");
}
