//! TBD
//! 

mod sealed {
    pub trait Sealed {}
}

/// Describes the handling behavior in case of allocation failure.
pub trait FailureHandling: sealed::Sealed + Send + Sync + Unpin {
    /// The type returned by allocating functions.
    /// 
    /// `Fallible` functions will return `Result<T, E>`,
    /// but `Fatal` functions will return `T`.
    type Result<T, E>;
}

/// Handle allocation failure globally by panicking / aborting.
#[derive(Debug)]
pub struct Fatal;

impl sealed::Sealed for Fatal {}
impl FailureHandling for Fatal {
    type Result<T, E> = T;
}

/// Handle allocation failure falliblyby returning a `Result`.
#[derive(Debug)]
pub struct Fallible;

impl sealed::Sealed for Fallible {}
impl FailureHandling for Fallible {
    type Result<T, E> = Result<T, E>;
}

/// Type parameter default `FailureHandling` for use in containers.
#[cfg(not(no_global_oom_handling))]
pub type DefaultFailureHandling = Fatal;
#[cfg(no_global_oom_handling)]
pub type DefaultFailureHandling = Fallible;
