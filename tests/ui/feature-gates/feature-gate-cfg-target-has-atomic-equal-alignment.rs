fn main() {
    cfg!(target_has_atomic_equal_alignment = "8");
    //~^ ERROR `cfg(target_has_atomic_equal_alignment)` is experimental and subject to change
    cfg!(target_has_atomic_equal_alignment = "16");
    //~^ ERROR `cfg(target_has_atomic_equal_alignment)` is experimental and subject to change
    cfg!(target_has_atomic_equal_alignment = "32");
    //~^ ERROR `cfg(target_has_atomic_equal_alignment)` is experimental and subject to change
    cfg!(target_has_atomic_equal_alignment = "64");
    //~^ ERROR `cfg(target_has_atomic_equal_alignment)` is experimental and subject to change
    cfg!(target_has_atomic_equal_alignment = "128");
    //~^ ERROR `cfg(target_has_atomic_equal_alignment)` is experimental and subject to change
    cfg!(target_has_atomic_equal_alignment = "ptr");
    //~^ ERROR `cfg(target_has_atomic_equal_alignment)` is experimental and subject to change
}
