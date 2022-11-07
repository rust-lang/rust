// check-pass
// edition:2021

#![allow(unreachable_code)]

async fn f(_: u8) {}

async fn g() {
    // Todo returns `!`, so the await is never reached, and in particular the
    // temporaries inside the formatting machinery are not still alive at the
    // await point.
    f(todo!("...")).await;
}

fn require_send(_: impl Send) {}

fn main() {
    require_send(g());
}
