//@ edition:2018
#![feature(must_not_suspend)]
#![deny(must_not_suspend)]

#[must_not_suspend = "You gotta use Umm's, ya know?"]
struct Umm {
    i: i64,
}

struct Bar {
    u: Umm,
}

async fn other() {}

impl Bar {
    async fn uhoh(&mut self) {
        let guard = &mut self.u; //~ ERROR `Umm` held across

        other().await;

        *guard = Umm { i: 2 }
    }
}

fn main() {}
