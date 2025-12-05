use crate::Directive;

#[derive(Debug)]
pub struct CkError {
    pub message: String,
    pub directive: Directive,
}
