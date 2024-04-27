pub const NAME_MAX: usize = {
    #[cfg(target_os = "linux")]
    {
        1024
    }
    #[cfg(target_os = "freebsd")]
    {
        255
    }
};
