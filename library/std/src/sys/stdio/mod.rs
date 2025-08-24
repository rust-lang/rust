#![forbid(unsafe_op_in_unsafe_fn)]

cfg_select! {
    any(target_family = "unix", target_os = "hermit") => {
        mod unix;
        pub use unix::*;
    }
    target_os = "windows" => {
        mod windows;
        pub use windows::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::*;
    }
    target_os = "solid_asp3" => {
        mod solid;
        pub use solid::*;
    }
    target_os = "teeos" => {
        mod teeos;
        pub use teeos::*;
    }
    target_os = "trusty" => {
        mod trusty;
        pub use trusty::*;
    }
    target_os = "uefi" => {
        mod uefi;
        pub use uefi::*;
    }
    target_os = "vexos" => {
        mod vexos;
        pub use vexos::*;
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod wasip1;
        pub use wasip1::*;
    }
    all(target_os = "wasi", target_env = "p2") => {
        mod wasip2;
        pub use wasip2::*;
    }
    target_os = "xous" => {
        mod xous;
        pub use xous::*;
    }
    target_os = "zkvm" => {
        mod zkvm;
        pub use zkvm::*;
    }
    _ => {
        mod unsupported;
        pub use unsupported::*;
    }
}
