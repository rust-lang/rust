use crate::FileId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CrateId(u32);

#[derive(Debug)]
pub struct Crate {
    root: FileId,
}

impl Crate {
    pub fn dependencies(&self) -> Vec<CrateId> {
        Vec::new()
    }
}
