#[unstable(feature = "uefi_std", issue = "none")]
impl From<crate::os::uefi::raw::Ipv6Address> for crate::net::Ipv6Addr {
    fn from(t: crate::os::uefi::raw::Ipv6Address) -> Self {
        Self::from(t.addr)
    }
}

#[unstable(feature = "uefi_std", issue = "none")]
impl From<&crate::net::Ipv6Addr> for crate::os::uefi::raw::Ipv6Address {
    fn from(t: &crate::net::Ipv6Addr) -> Self {
        Self { addr: t.octets() }
    }
}

#[unstable(feature = "uefi_std", issue = "none")]
impl From<crate::os::uefi::raw::Ipv4Address> for crate::net::Ipv4Addr {
    fn from(t: crate::os::uefi::raw::Ipv4Address) -> Self {
        Self::from(t.addr)
    }
}

#[unstable(feature = "uefi_std", issue = "none")]
impl From<&crate::net::Ipv4Addr> for crate::os::uefi::raw::Ipv4Address {
    fn from(t: &crate::net::Ipv4Addr) -> Self {
        Self { addr: t.octets() }
    }
}
