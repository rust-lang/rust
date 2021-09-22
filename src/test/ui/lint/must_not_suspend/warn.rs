// edition:2018
// run-pass
#![feature(must_not_suspend)]

#[must_not_suspend = "You gotta use Umm's, ya know?"]
struct Umm {
    _i: i64
}


fn bar() -> Umm {
    Umm {
        _i: 1
    }
}

async fn other() {}

pub async fn uhoh() {
    let _guard = bar(); //~ WARNING `Umm` held across
    other().await;
}

fn main() {
}
