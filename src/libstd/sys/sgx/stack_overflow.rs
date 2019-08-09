pub struct Handler;

impl Handler {
    pub unsafe fn new() -> Handler {
        Handler
    }
}

#[cfg_attr(test, allow(dead_code))]
pub unsafe fn init() {
}

pub unsafe fn cleanup() {
}
