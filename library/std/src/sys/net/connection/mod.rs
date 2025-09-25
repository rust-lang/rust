cfg_select! {
    any(
        all(target_family = "unix", not(target_os = "l4re")),
        target_os = "windows",
        target_os = "hermit",
        all(target_os = "wasi", target_env = "p2"),
        target_os = "solid_asp3",
    ) => {
        mod socket;
        pub use socket::*;
    }
    all(target_vendor = "fortanix", target_env = "sgx") => {
        mod sgx;
        pub use sgx::*;
    }
    all(target_os = "wasi", target_env = "p1") => {
        mod wasip1;
        pub use wasip1::*;
    }
    target_os = "xous" => {
        mod xous;
        pub use xous::*;
    }
    target_os = "uefi" => {
        mod uefi;
        pub use uefi::*;
    }
    _ => {
        mod unsupported;
        pub use unsupported::*;
    }
}

#[cfg_attr(
    // Make sure that this is used on some platforms at least.
    not(any(target_os = "linux", target_os = "windows")),
    allow(dead_code)
)]
fn each_addr<A: crate::net::ToSocketAddrs, F, T>(addr: A, mut f: F) -> crate::io::Result<T>
where
    F: FnMut(&crate::net::SocketAddr) -> crate::io::Result<T>,
{
    use crate::io::Error;

    let mut last_err = None;
    for addr in addr.to_socket_addrs()? {
        match f(&addr) {
            Ok(l) => return Ok(l),
            Err(e) => last_err = Some(e),
        }
    }

    match last_err {
        Some(err) => Err(err),
        None => Err(Error::NO_ADDRESSES),
    }
}
