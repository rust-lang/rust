use crate::ffi::OsString;
use crate::vec::IntoIter;

pub type Args = IntoIter<OsString>;

pub fn args() -> Args {
    Vec::new().into_iter()
}
