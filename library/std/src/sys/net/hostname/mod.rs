cfg_select! {
    target_family = "unix" => {
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
