cfg_select! {
    any(
        all(target_family = "unix", not(target_os = "l4re")),
        target_os = "windows",
        target_os = "hermit",
        all(target_os = "wasi", target_env = "p2"),
        target_os = "solid_asp3",
    ) => {
        mod connection {
            mod socket;
            pub use socket::*;
        }
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod connection {
            mod sgx;
            pub use sgx::*;
        }
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod connection {
            mod wasip1;
            pub use wasip1::*;
        }
    }
    target_os = "xous" => {
        mod connection {
            mod xous;
            pub use xous::*;
        }
    }
    target_os = "uefi" => {
        mod connection {
            mod uefi;
            pub use uefi::*;
        }
    }
    _ => {
        mod connection {
            mod unsupported;
            pub use unsupported::*;
        }
    }
}

pub use connection::*;
