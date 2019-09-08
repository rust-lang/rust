use std::{cell::Cell, fmt};

use ra_db::{CrateId, FileId};

use crate::{db::HirDatabase, Crate, Module, Name};

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
        let path = self.file_path(file_id);
        fmt.debug_struct("Module")
            .field("name", &module.name(self).unwrap_or_else(Name::missing))
            .field("path", &path.unwrap_or_else(|| "N/A".to_string()))
            .finish()
    }
}

fn debug_fn(f: impl FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result) -> impl fmt::Debug {
    struct DebugFn<F>(Cell<Option<F>>);

    impl<F: FnOnce(&mut fmt::Formatter<'_>) -> fmt::Result> fmt::Debug for DebugFn<F> {
        fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
            let f = self.0.take().unwrap();
            f(fmt)
        }
    }

    DebugFn(Cell::new(Some(f)))
}
