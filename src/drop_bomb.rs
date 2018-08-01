use std::borrow::Cow;

pub struct DropBomb {
    msg: Cow<'static, str>,
    defused: bool,
}

impl DropBomb {
    pub fn new(msg: impl Into<Cow<'static, str>>) -> DropBomb {
        DropBomb { msg: msg.into(), defused: false }
    }
    pub fn defuse(&mut self) { self.defused = true }
}

impl Drop for DropBomb {
    fn drop(&mut self) {
        if !self.defused && !::std::thread::panicking() {
            panic!("{}", self.msg)
        }
    }
}
