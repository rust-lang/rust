cfg_if::cfg_if! {
    if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod sgx;
        pub use sgx::RwLock;
    } else if #[cfg(target_os = "solid_asp3")] {
        mod solid;
        pub use solid::RwLock;
    } else if #[cfg(target_os = "teeos")] {
        mod teeos;
        pub use teeos::RwLock;
    }
}
