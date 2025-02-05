cfg_if::cfg_if! {
    if #[cfg(any(
        all(target_family = "unix", not(target_os = "l4re")),
        target_os = "windows",
        target_os = "hermit",
        all(target_os = "wasi", target_env = "p2"),
        target_os = "solid_asp3",
    ))] {
        mod connection {
            mod socket;
            pub use socket::*;
        }
    } else if #[cfg(all(target_vendor = "fortanix", target_env = "sgx"))] {
        mod connection {
            mod sgx;
            pub use sgx::*;
        }
    } else if #[cfg(all(target_os = "wasi", target_env = "p1"))] {
        mod connection {
            mod wasip1;
            pub use wasip1::*;
        }
    } else if #[cfg(target_os = "xous")] {
        mod connection {
            mod xous;
            pub use xous::*;
        }
    } else {
        mod connection {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

pub use connection::*;
