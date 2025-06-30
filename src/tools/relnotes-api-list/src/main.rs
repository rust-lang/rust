use crate::convert_to_schema::ConvertToSchema;
use crate::schema::{CURRENT_SCHEMA_VERSION, Schema, SchemaItem};
use crate::stability::StabilityStore;
use crate::store::Store;
use crate::url::UrlStore;
use crate::visitor::Visitor;
use anyhow::{Context as _, Error};

mod convert_to_schema;
mod pretty_print;
mod schema;
mod stability;
mod store;
mod url;
mod visitor;

// There is no need to include `core` or `alloc` here, as their content is re-exported in the `std`
// crate. If we were to include them in the list, we would have all items added to `core` duplicated
// between `core` and `std`.
static CRATES_TO_EMIT: &[&str] = &["std", "test", "proc_macro"];

fn main() -> Result<(), Error> {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    let [rustdoc_json_dir, dest] = args.as_slice() else {
        eprintln!("usage: relnotes-api-list <rustdoc-json-dir> <dest>");
        std::process::exit(1)
    };

    let mut store = Store::new();
    let mut all_crates = Vec::new();
    let mut crates_to_emit = Vec::new();
    for entry in std::fs::read_dir(rustdoc_json_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let id = store.load(&path).with_context(|| format!("failed to parse {path:?}"))?;
        all_crates.push(id);
        if CRATES_TO_EMIT.contains(&store.name_of_crate(id)) {
            crates_to_emit.push(id);
        }
    }

    let mut stability = StabilityStore::new();
    for id in &all_crates {
        stability.add(&store, &store.crate_root(*id)?).with_context(|| {
            format!("failed to gather stability for {}", store.name_of_crate(*id))
        })?;
    }

    let urls = UrlStore::new(&store, &all_crates).context("failed to calculate URLs")?;

    let mut result = Schema { schema_version: CURRENT_SCHEMA_VERSION, items: Vec::new() };
    for id in crates_to_emit {
        result.items.extend(
            ConvertToSchema::new(&store, &stability, &urls)
                .visit_item(&store.crate_root(id)?)
                .with_context(|| format!("failed to process crate {}", store.name_of_crate(id)))?,
        );
    }
    std::fs::write(dest, &serde_json::to_string_pretty(&result)?)?;

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
