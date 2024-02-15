cfg_if::cfg_if! {
    if #[cfg(target_os = "solid_asp3")] {
        mod solid;
        pub use solid::RwLock;
    }
}
