use anyhow::{Context as _, Error};

use crate::check_urls::check_urls;
use crate::convert_to_schema::ConvertToSchema;
use crate::schema::{CURRENT_SCHEMA_VERSION, Schema, SchemaItem};
use crate::stability::StabilityStore;
use crate::store::Store;
use crate::visitor::Visitor;

mod check_urls;
mod convert_to_schema;
mod pretty_print;
mod schema;
mod stability;
mod store;
mod utils;
mod visitor;

/// List of crates that will be included in the API list. Note that this also affects URL
/// generation: removing crates we generate documentation for, or adding crates we don't, will
/// result in broken URLs.
static PUBLIC_CRATES: &[&str] = &["core", "alloc", "std", "test", "proc_macro"];

fn main() -> Result<(), Error> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [rustdoc_json_dir, dest] = args.as_slice() else {
        eprintln!("usage: relnotes-api-list <rustdoc-json-dir> <dest>");
        std::process::exit(1)
    };

    let mut store = Store::new();
    let mut crates_to_emit = Vec::new();
    for entry in std::fs::read_dir(rustdoc_json_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let id = store.load(&path).with_context(|| format!("failed to parse {path:?}"))?;
        if PUBLIC_CRATES.contains(&store.crate_name(id)) {
            crates_to_emit.push(id);
        }
    }

    let mut stability = StabilityStore::new();
    for id in store.crate_ids() {
        stability
            .add(&store, &store.crate_root(id)?)
            .with_context(|| format!("failed to gather stability for {}", store.crate_name(id)))?;
    }

    let mut result = Schema { schema_version: CURRENT_SCHEMA_VERSION, items: Vec::new() };
    for id in crates_to_emit {
        result.items.extend(
            ConvertToSchema::new(&store, &stability)?
                .visit_item(&store.crate_root(id)?)
                .with_context(|| format!("failed to process crate {}", store.crate_name(id)))?,
        );
    }
    std::fs::write(dest, &serde_json::to_string_pretty(&result)?)?;

    check_urls(&result)?;

    eprintln!("found {} documentation items", count_items(&result.items));

    Ok(())
}

fn count_items(items: &[SchemaItem]) -> usize {
    let mut count = 0;
    for item in items {
        count += 1;
        count += count_items(&item.children);
    }
    count
}
