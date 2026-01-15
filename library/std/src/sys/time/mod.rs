cfg_select! {
    target_os = "motor" => {
        use moto_rt::time as imp;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        use sgx as imp;
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
