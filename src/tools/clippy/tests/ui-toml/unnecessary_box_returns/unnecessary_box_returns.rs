#![warn(clippy::unnecessary_box_returns)]

fn f() -> Box<[u8; 64]> {
    //~^ ERROR: boxed return of the sized type `[u8; 64]`
    todo!()
}
fn f2() -> Box<[u8; 65]> {
    todo!()
}

fn main() {}
