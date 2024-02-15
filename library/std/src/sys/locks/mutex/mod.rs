cfg_if::cfg_if! {
    if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::Mutex;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod itron;
        pub use itron::Condvar;
    }
}
