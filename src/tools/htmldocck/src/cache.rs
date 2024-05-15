use std::{
    collections::{hash_map::Entry, HashMap},
    path::Path,
};

use crate::error::DiagCtxt;

pub(crate) struct Cache<'a> {
    root: &'a Path,
    // FIXME: `&'a str`s
    files: HashMap<String, String>,
    // FIXME: `&'a str`, comment what this is for -- `-`
    last_path: Option<String>,
}

impl<'a> Cache<'a> {
    pub(crate) fn new(root: &'a Path) -> Self {
        Self { root, files: HashMap::new(), last_path: None }
    }

    // FIXME: check file vs. dir (`@has <PATH>` vs. `@has-dir <PATH>`)
    /// Check if the path points to an existing entity.
    pub(crate) fn has(&mut self, path: String, dcx: &mut DiagCtxt) -> Result<bool, ()> {
        // FIXME: should we use `try_exists` over `exists` instead? matters the most for `@!has <PATH>`.
        let path = self.resolve(path, dcx)?;

        Ok(self.files.contains_key(&path) || Path::new(&path).exists())
    }

    /// Load the contents of the given path.
    pub(crate) fn load(&mut self, path: String, dcx: &mut DiagCtxt) -> Result<&str, ()> {
        let path = self.resolve(path, dcx)?;

        Ok(match self.files.entry(path) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // FIXME: better message, location
                let data =
                    std::fs::read_to_string(self.root.join(entry.key())).map_err(|error| {
                        dcx.emit(&format!("failed to read file: {error}"), None, None)
                    })?;
                entry.insert(data)
            }
        })
    }

    // FIXME: &str -> &str if possible
    fn resolve(&mut self, path: String, dcx: &mut DiagCtxt) -> Result<String, ()> {
        if path == "-" {
            // FIXME: no cloning
            return self
                .last_path
                .clone()
                // FIXME better diag, location
                .ok_or_else(|| {
                    dcx.emit(
                        "attempt to use `-` ('previous path') in the very first command",
                        None,
                        None,
                    )
                });
        }

        // While we could normalize the `path` at this point by
        // using `std::path::absolute`, it's likely not worth it.
        self.last_path = Some(path.clone());
        Ok(path)
    }
}
