#![allow(dead_code)]
use tester::Window;

macro_rules! info {
    ($s:literal, $x:expr) => {{
        let _ = $x;
    }};
}

struct WindowState {
    window: Window,
}

impl WindowState {
    fn takes_ref(&self) {
        info!("{:?}", self.window.id());
    }

    fn takes_mut(&mut self) {
        info!("{:?}", self.window.id());
    }
}

fn main() {}
