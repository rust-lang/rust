//@ edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend = "You gotta use Umm's, ya know?"]
struct Umm {
    i: i64
}


fn bar() -> Box<Umm> {
    Box::new(Umm {
        i: 1
    })
}

async fn other() {}

pub async fn uhoh() {
    let _guard = bar(); //~ ERROR boxed `Umm` held across
    other().await;
}

fn main() {
}
