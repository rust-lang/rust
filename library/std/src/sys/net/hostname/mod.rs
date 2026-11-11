cfg_select! {
    all(target_family = "unix", not(target_os = "espidf")) => {
        mod unix;
        pub use unix::hostname;
    }
    // `GetHostNameW` is only available starting with Windows 8.
    all(target_os = "windows", not(target_vendor = "win7")) => {
        mod windows;
        pub use windows::hostname;
    }
    _ => {
        mod unsupported;
        pub use unsupported::hostname;
    }
}
