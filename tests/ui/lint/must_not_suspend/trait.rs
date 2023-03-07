// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
// edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend]
trait Wow {}

impl Wow for i32 {}

fn r#impl() -> impl Wow {
    1
}

fn r#dyn() -> Box<dyn Wow> {
    Box::new(1)
}

async fn other() {}

pub async fn uhoh() {
    let _guard1 = r#impl(); //~ ERROR implementer of `Wow` held across
    let _guard2 = r#dyn(); //~ ERROR boxed `Wow` trait object held across

    other().await;

    drop(_guard1);
    drop(_guard2);
}

fn main() {
}
