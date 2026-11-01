#[derive(Clone)]
pub struct Custom;

impl Custom {
    pub fn wake(self) {}
}

macro_rules! mac {
    ($cx:ident) => {
        $cx.waker()
    };
}

pub fn wake(cx: &mut std::task::Context) {
    cx.waker().clone().wake();
    //~^ waker_clone_wake

    mac!(cx).clone().wake();
    //~^ waker_clone_wake
}

pub fn no_lint(cx: &mut std::task::Context, c: &Custom) {
    c.clone().wake();

    let w = cx.waker().clone();
    w.wake();

    cx.waker().clone().wake_by_ref();
}

fn main() {}
