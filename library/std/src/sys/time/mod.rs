cfg_select! {
    target_os = "hermit" => {
        mod hermit;
        use hermit as imp;
    }
    target_os = "motor" => {
        use moto_rt::time as imp;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        use sgx as imp;
    }
    target_os = "solid_asp3" => {
        mod solid;
        use solid as imp;
    }
    target_os = "uefi" => {
        mod uefi;
        use uefi as imp;
    }
    any(
        target_os = "teeos",
        target_family = "unix",
        target_os = "wasi",
    ) => {
        mod unix;
        use unix as imp;
    }
    target_os = "vexos" => {
        mod vexos;
        #[expect(unused)]
        mod unsupported;

        mod imp {
            pub use super::vexos::Instant;
            pub use super::unsupported::{SystemTime, UNIX_EPOCH};
        }
    }
    target_os = "windows" => {
        mod windows;
        use windows as imp;
    }
    target_os = "xous" => {
        mod xous;
        use xous as imp;
    }
    _ => {
        mod unsupported;
        use unsupported as imp;
    }
}

pub use imp::{Instant, SystemTime, UNIX_EPOCH};
