/// Used as a return value to signify a fatal error occurred. (It is also
/// used as the argument to panic at the moment, but that will eventually
/// not be true.)
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct FatalError;

pub struct FatalErrorMarker;

// Don't implement Send on FatalError. This makes it impossible to panic!(FatalError).
// We don't want to invoke the panic handler and print a backtrace for fatal errors.
impl !Send for FatalError {}

impl FatalError {
    pub fn raise(self) -> ! {
        std::panic::resume_unwind(Box::new(FatalErrorMarker))
    }
}

impl std::fmt::Display for FatalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parser fatal error")
    }
}

impl std::error::Error for FatalError {}
