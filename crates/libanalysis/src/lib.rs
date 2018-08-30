extern crate failure;
extern crate parking_lot;
#[macro_use]
extern crate log;
extern crate once_cell;
extern crate libsyntax2;
extern crate libeditor;
extern crate fst;
extern crate rayon;
extern crate relative_path;

mod symbol_index;
mod module_map;
mod api;
mod imp;

pub use self::symbol_index::Query;
pub use self::api::{
    AnalysisHost, Analysis, SourceChange, SourceFileEdit, FileSystemEdit, Position, Diagnostic, Runnable, RunnableKind,
    FileId, FileResolver,
};

pub type Result<T> = ::std::result::Result<T, ::failure::Error>;
