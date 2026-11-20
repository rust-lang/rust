use rustc_macros::StableHash;

/// This error is emitted by comptime files queries when a file cannot be read.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[derive(StableHash)]
pub enum ReadFileError<'tcx> {
    /// Another IO error happened while trying to read the file
    IO(&'tcx str),

    /// The file being accessed at comptime has not been whitelisted
    NotWhitelisted,
}
