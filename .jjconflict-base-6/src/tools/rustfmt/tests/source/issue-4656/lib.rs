extern crate cfg_if;

cfg_if::cfg_if! {
    if #[cfg(target_family = "unix")] {
        mod format_me_please;
    }
}
