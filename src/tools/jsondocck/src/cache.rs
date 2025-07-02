use std::borrow::Cow;
use std::collections::HashMap;
use std::iter;
use std::path::{Path, PathBuf};

use fs_err as fs;
use jaq_core::load::{Arena, File, Loader};
use jaq_core::{Compiler, Ctx, RcIter};
use jaq_json::Val;
use serde_json::Value;

use crate::config::Config;

#[derive(Debug)]
pub struct Cache {
    value: Val,
    global_vars: HashMap<String, Val>,
}

impl Cache {
    /// Create a new cache, used to read a documentation JSON file only once and otherwise store its
    /// content.
    pub fn new(Config { doc_dir, template }: &Config) -> Self {
        let root = Path::new(doc_dir);
        // `filename` needs to replace `-` with `_` to be sure the JSON path will always be valid.
        let mut filename: PathBuf =
            Path::new(template).file_stem().unwrap().to_str().unwrap().replace('-', "_").into();

        filename.set_extension("json");

        let content = fs::read(root.join(filename)).expect("failed to read a JSON file");

        Cache {
            value: serde_json::from_slice::<Value>(&content)
                .expect("failed to convert from JSON")
                .into(),
            // FIXME: Replace this with empty `HashMap` and use `input_filename` instead of `$FILE`
            // in tests once https://github.com/01mf02/jaq/issues/144 is fixed.
            global_vars: [("$FILE".into(), template.to_owned().into())].into(),
        }
    }

    pub fn arg(&mut self, name: &str, value: Val) {
        self.global_vars.insert(format!("${name}"), value);
    }

    pub fn filter(&self, code: &str) -> Result<Val, Cow<'static, str>> {
        let arena = Arena::default();
        let modules = Loader::new(
            // `tonumber` depends on `fromjson` that requires adding the extra "hifijson"
            // dependency.
            jaq_std::defs().chain(jaq_json::defs().filter(|def| !matches!(def.name, "tonumber"))),
        )
        .load(&arena, File { code, path: () })
        .map_err(|e| format!("failed to parse the given filter: {:?}", e.first().unwrap().1))?;
        let filter = Compiler::default()
            .with_funs(jaq_std::funs().chain(jaq_json::base_funs()))
            .with_global_vars(self.global_vars.keys().map(String::as_ref))
            .compile(modules)
            .map_err(|e| {
                format!("failed to compile the given filter: {:?}", e.first().unwrap().1)
            })?;
        let inputs = RcIter::new(iter::empty());
        let mut outputs =
            filter.run((Ctx::new(self.global_vars.values().cloned(), &inputs), self.value.clone()));
        let output = outputs
            .next()
            .ok_or::<Cow<'_, _>>("directive returned nothing".into())?
            .map_err(|e| format!("failed to execute the given filter: {e}"))?;

        if outputs.next().is_some() {
            return Err("expected a single output, received multiple".into());
        }

        Ok(output)
    }
}
