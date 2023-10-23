#[derive(Clone)]
pub struct Custom;

impl Custom {
    pub fn wake(self) {}
}

pub fn wake(cx: &mut std::task::Context) {
    cx.waker().clone().wake();

    // We don't do that for now
    let w = cx.waker().clone();
    w.wake();

    cx.waker().clone().wake_by_ref();
}

pub fn no_lint(c: &Custom) {
    c.clone().wake()
}

fn main() {}
