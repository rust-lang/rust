#![crate_name = "issue_86620_1"]

pub trait VZip {
    fn vzip() -> usize;
}

impl<T> VZip for T {
    fn vzip() -> usize {
        0
    }
}
