fn main() {
    cfg!(target_has_atomic_load_store = "8");
    //~^ ERROR `cfg(target_has_atomic_load_store)` is experimental and subject to change
    cfg!(target_has_atomic_load_store = "16");
    //~^ ERROR `cfg(target_has_atomic_load_store)` is experimental and subject to change
    cfg!(target_has_atomic_load_store = "32");
    //~^ ERROR `cfg(target_has_atomic_load_store)` is experimental and subject to change
    cfg!(target_has_atomic_load_store = "64");
    //~^ ERROR `cfg(target_has_atomic_load_store)` is experimental and subject to change
    cfg!(target_has_atomic_load_store = "128");
    //~^ ERROR `cfg(target_has_atomic_load_store)` is experimental and subject to change
    cfg!(target_has_atomic_load_store = "ptr");
    //~^ ERROR `cfg(target_has_atomic_load_store)` is experimental and subject to change
}
