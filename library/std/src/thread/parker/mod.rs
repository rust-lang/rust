cfg_if::cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "android"))] {
        mod linux;
        pub use linux::Parker;
    } else {
        mod generic;
        pub use generic::Parker;
    }
}
