cfg_select! {
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
    target_os = "vexos" => {
        mod vexos;
        #[expect(unused)]
        mod unsupported;

        mod imp {
            pub use super::vexos::Instant;
            pub use super::unsupported::{SystemTime, UNIX_EPOCH};
        }
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
