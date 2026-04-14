pub mod ids {
    include!(concat!(env!("OUT_DIR"), "/pci_ids_gen.rs"));
}

pub const NAME_SOURCE: &str = ids::PCI_IDS_NAME_SOURCE;

#[inline]
pub fn lookup_names(vendor: u16, device: u16) -> (Option<&'static str>, Option<&'static str>) {
    (ids::vendor_name(vendor), ids::device_name(vendor, device))
}

pub fn fmt_pci_id(vendor: u16, device: u16) -> PciIdDisplay {
    PciIdDisplay { vendor, device }
}

pub struct PciIdDisplay {
    vendor: u16,
    device: u16,
}

impl core::fmt::Display for PciIdDisplay {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let v = ids::vendor_name(self.vendor);
        let d = ids::device_name(self.vendor, self.device);
        match (v, d) {
            (Some(vn), Some(dn)) => write!(f, "{} {}", vn, dn),
            (Some(vn), None) => write!(f, "{} device={:04x}", vn, self.device),
            _ => write!(f, "(unknown vendor) (unknown device)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn display_fallbacks() {
        // The generated table is not available at test time here, so we just
        // exercise the fallback path without crashing.
        let s = fmt_pci_id(0x1234, 0xabcd).to_string();
        assert!(s.contains("unknown vendor"));
    }
}
