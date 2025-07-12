use std::borrow::Cow;
use std::iter::{self, Empty};
use std::path::{Path, PathBuf};

use foldhash::fast::RandomState;
use fs_err as fs;
use indexmap::IndexMap as InnerIndexMap;
use jaq_core::load::{Arena, File, Loader};
use jaq_core::{Compiler, Ctx, Filter as InnerFilter, Native, RcIter};
use jaq_json::Val;
use serde_json::Value;

use crate::config::Config;

type IndexMap<K, V> = InnerIndexMap<K, V, RandomState>;

#[derive(Debug)]
pub struct Cache {
    value: Val,
    global_vars: IndexMap<String, Val>,
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
            global_vars: [("$FILE".into(), template.to_owned().into())].into_iter().collect(),
        }
    }

    pub fn arg(&mut self, name: &str, value: Val) {
        self.global_vars.insert(format!("${name}"), value);
    }

    pub fn filter(&self, code: &str) -> Result<Filter, Cow<'static, str>> {
        let arena = Arena::default();
        let modules = Loader::new(
            // `tonumber` depends on `fromjson` that requires adding the extra "hifijson"
            // dependency.
            jaq_std::defs().chain(jaq_json::defs().filter(|def| !matches!(def.name, "tonumber"))),
        )
        .load(&arena, File { code, path: () })
        .map_err(|e| format!("failed to parse the given filter: {e:?}"))?;

        Ok(Filter {
            inner: Compiler::default()
                .with_funs(jaq_std::funs().chain(jaq_json::base_funs()))
                .with_global_vars(self.global_vars.keys().map(String::as_ref))
                .compile(modules)
                .map_err(|e| format!("failed to compile the given filter: {e:?}"))?,
            inputs: RcIter::new(iter::empty()),
        })
    }
}

pub struct Filter {
    inner: InnerFilter<Native<Val>>,
    inputs: RcIter<Empty<Result<Val, String>>>,
}

impl Filter {
    pub fn run(&self, cache: &Cache) -> Values<impl Iterator<Item = Result<Val, String>> + '_> {
        Values {
            inner: self
                .inner
                .run((
                    Ctx::new(cache.global_vars.values().cloned(), &self.inputs),
                    cache.value.clone(),
                ))
                .map(|r| r.map_err(|e| format!("failed to execute the given filter: {e}"))),
            counter: 0,
        }
    }
}

pub struct Values<I> {
    inner: I,
    counter: usize,
}

impl<I: Iterator<Item = Result<Val, String>>> Values<I> {
    pub fn is_empty(&mut self) -> Result<(), String> {
        if self.inner.next().is_none() {
            Ok(())
        } else {
            Err(format!(
                "expected {expected} value(s), received more than {expected}",
                expected = self.counter
            ))
        }
    }

    pub fn next(&mut self) -> Result<Val, String> {
        self.counter += 1;

        self.inner
            .next()
            .ok_or_else(|| format!("{} value was expected, received nothing", self.counter))?
    }
}
