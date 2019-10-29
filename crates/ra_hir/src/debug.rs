//! printf debugging infrastructure for rust-analyzer.
//!
//! When you print a hir type, like a module, using `eprintln!("{:?}", module)`,
//! you usually get back a numeric ID, which doesn't tell you much:
//! `Module(92)`.
//!
//! This module adds convenience `debug` methods to various types, which resolve
//! the id to a human-readable location info:
//!
//! ```not_rust
//! eprintln!("{:?}", module.debug(db));
//! =>
//! Module { name: collections, path: "liballoc/collections/mod.rs" }
//! ```
//!
//! Note that to get this info, we might need to execute queries! So
//!
//! * don't use the `debug` methods for logging
//! * when debugging, be aware that interference is possible.

use std::fmt;

use ra_db::{CrateId, FileId};

use crate::{db::HirDatabase, Crate, HirFileId, Module, Name};

impl Crate {
    pub fn debug(self, db: &impl HirDebugDatabase) -> impl fmt::Debug + '_ {
        debug_fn(move |fmt| db.debug_crate(self, fmt))
    }
}

impl Module {
    pub fn debug(self, db: &impl HirDebugDatabase) -> impl fmt::Debug + '_ {
        debug_fn(move |fmt| db.debug_module(self, fmt))
    }
}

pub trait HirDebugHelper: HirDatabase {
    fn crate_name(&self, _krate: CrateId) -> Option<String> {
        None
    }
    fn file_path(&self, _file_id: FileId) -> Option<String> {
        None
    }
}

pub trait HirDebugDatabase {
    fn debug_crate(&self, krate: Crate, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn debug_module(&self, module: Module, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;
    fn debug_hir_file_id(&self, file_id: HirFileId, fmt: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<DB: HirDebugHelper> HirDebugDatabase for DB {
    fn debug_crate(&self, krate: Crate, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut builder = fmt.debug_tuple("Crate");
        match self.crate_name(krate.crate_id) {
            Some(name) => builder.field(&name),
            None => builder.field(&krate.crate_id),
        }
        .finish()
    }

    fn debug_module(&self, module: Module, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file_id = module.definition_source(self).file_id.original_file(self);
        let path = self.file_path(file_id).unwrap_or_else(|| "N/A".to_string());
        fmt.debug_struct("Module")
            .field("name", &module.name(self).unwrap_or_else(Name::missing))
            .field("path", &path)
            .finish()
    }

    fn debug_hir_file_id(&self, file_id: HirFileId, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let original = file_id.original_file(self);
        let path = self.file_path(original).unwrap_or_else(|| "N/A".to_string());
        let is_macro = file_id != original.into();
        fmt.debug_struct("HirFileId").field("path", &path).field("macro", &is_macro).finish()
    }
}

fn debug_fn(f: impl Fn(&mut fmt::Formatter<'_>) -> fmt::Result) -> impl fmt::Debug {
    struct DebugFn<F>(F);

    impl<F: Fn(&mut fmt::Formatter<'_>) -> fmt::Result> fmt::Debug for DebugFn<F> {
        fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            (&self.0)(fmt)
        }
    }

    DebugFn(f)
}
