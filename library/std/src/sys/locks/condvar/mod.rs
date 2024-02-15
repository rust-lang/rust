cfg_if::cfg_if! {
    if #[cfg(target_os = "solid_asp3")] {
        mod itron;
        pub use itron::Condvar;
    }
}
