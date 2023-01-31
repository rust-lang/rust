// edition:2018
// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir

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

        let _g = &*guard;
        *guard = Umm { i: 2 }
    }
}

fn main() {}
