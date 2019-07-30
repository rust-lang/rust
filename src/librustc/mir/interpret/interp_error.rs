//! macros to do something like `.ok_or_else(|| inval!(TooGeneric).into())` rather than
//! `.ok_or_else(|| InterpError::InvalidProgram(TooGeneric).into())`

#[macro_export]
macro_rules! inval {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::InvalidProgram(
            $crate::mir::interpret::InvalidProgramInfo::$($tt)*
        )
    };
}

#[macro_export]
macro_rules! unsup {
    ($($tt:tt)*) => {
        $crate::mir::interpret::InterpError::Unsupported(
            $crate::mir::interpret::UnsupportedOpInfo::$($tt)*
        )
    };
}
