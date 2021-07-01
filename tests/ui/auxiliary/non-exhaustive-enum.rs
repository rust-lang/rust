// Stripped down version of the ErrorKind enum of std
#[non_exhaustive]
pub enum ErrorKind {
    NotFound,
    PermissionDenied,
    #[doc(hidden)]
    Uncategorized,
}
