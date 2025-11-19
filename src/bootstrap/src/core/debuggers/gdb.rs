use std::path::Path;

use crate::core::builder::Builder;

pub(crate) struct Gdb<'a> {
    pub(crate) gdb: &'a Path,
}

pub(crate) fn discover_gdb<'a>(builder: &'a Builder<'_>) -> Option<Gdb<'a>> {
    let gdb = builder.config.gdb.as_deref()?;

    Some(Gdb { gdb })
}
