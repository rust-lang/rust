//@ check-pass
//@ run-rustfix

#[warn(unreachable_pub)]
mod inner {
    #[allow(unused)]
    pub enum T {
        //~^ WARN unreachable `pub` item
        A(u8),
        X { a: f32, b: () },
    }
}

fn main() {}
