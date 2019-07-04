// build-pass (FIXME(62277): could be check-pass?)
// note this was only reproducible with lib crates
// compile-flags: --crate-type=lib

pub struct Hz;

impl Hz {
    pub const fn num(&self) -> u32 {
        42
    }
    pub const fn normalized(&self) -> Hz {
        Hz
    }

    pub const fn as_u32(&self) -> u32 {
        self.normalized().num() // this used to promote the `self.normalized()`
    }
}
