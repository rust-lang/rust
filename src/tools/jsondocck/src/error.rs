use crate::Command;

#[derive(Debug)]
pub struct CkError {
    pub message: String,
    pub command: Command,
}
