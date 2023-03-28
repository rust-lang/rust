// derived from miniz_oxide

pub enum MZStatus {}
pub enum MZError {}

pub struct StreamResult;

pub type MZResult = Result<MZStatus, MZError>;

impl core::convert::From<StreamResult> for MZResult {
    fn from(res: StreamResult) -> Self {
        loop {}
    }
}

impl core::convert::From<&StreamResult> for MZResult {
    fn from(res: &StreamResult) -> Self {
        loop {}
    }
}
