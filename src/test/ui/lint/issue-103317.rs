// check-pass

#[warn(unreachable_pub)]
#[allow(unused)]
mod inner {
    pub enum T {
        //~^ WARN unreachable `pub` item
        A(u8),
        X { a: f32, b: () },
    }
}

fn main() {}
