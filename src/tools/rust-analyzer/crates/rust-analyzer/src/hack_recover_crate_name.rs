//! Currently cargo does not emit crate name in the `cargo test --format=json`, which needs to be changed. This
//! module contains a way to recover crate names in a very hacky and wrong way.

// FIXME(hack_recover_crate_name): Remove this module.

use std::sync::{Mutex, MutexGuard, OnceLock};

use ide_db::FxHashMap;

static STORAGE: OnceLock<Mutex<FxHashMap<String, String>>> = OnceLock::new();

fn get_storage() -> MutexGuard<'static, FxHashMap<String, String>> {
    STORAGE.get_or_init(|| Mutex::new(FxHashMap::default())).lock().unwrap()
}

pub(crate) fn insert_name(name_with_crate: String) {
    let Some((_, name_without_crate)) = name_with_crate.split_once("::") else {
        return;
    };
    get_storage().insert(name_without_crate.to_owned(), name_with_crate);
}

pub(crate) fn lookup_name(name_without_crate: String) -> Option<String> {
    get_storage().get(&name_without_crate).cloned()
}
