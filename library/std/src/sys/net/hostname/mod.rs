cfg_select! {
    all(target_family = "unix", not(target_os = "espidf")) => {
        mod unix;
        pub use unix::hostname;
    }
    target_os = "windows" => {
        mod windows;
        pub use windows::hostname;
    }
    _ => {
        mod unsupported;
        pub use unsupported::hostname;
    }
}
