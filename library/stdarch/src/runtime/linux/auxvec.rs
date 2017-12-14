//! Reads /proc/self/auxv on Linux systems

use std::prelude::v1::*;
use std::slice;
use std::mem;

/// Simple abstraction for the ELF Auxiliary Vector
///
/// the elf.h provide the layout of the single entry as auxv_t.
/// The desugared version is a usize tag followed by a union with
/// the same storage size.
///
/// Cache only the HWCAP and HWCAP2 entries.
#[derive(Debug)]
pub struct AuxVec {
    hwcap: Option<usize>,
    hwcap2: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
/// ELF Auxiliary vector entry types
///
/// The entry types are specified in  [linux/auxvec.h][auxvec_h].
///
/// [auxvec_h]: https://github.com/torvalds/linux/blob/master/include/uapi/linux/auxvec.h
pub enum AT {
    /// CPU Hardware capabilities, it is a bitfield.
    HWCAP = 16,
    /// CPU Hardware capabilities, additional bitfield.
    HWCAP2 = 26,
}

impl AuxVec {
    /// Reads the ELF Auxiliary Vector
    ///
    /// Try to read `/proc/self/auxv`.
    // TODO: Make use of getauxval once it is available in a
    // reliable way.
    pub fn new() -> Result<Self, ::std::io::Error> {
        use std::io::Read;
        let mut file = ::std::fs::File::open("/proc/self/auxv")?;
        let mut buf = [0usize; 64];
        let mut raw = unsafe {
            slice::from_raw_parts_mut(
                buf.as_mut_ptr() as *mut u8,
                buf.len() * mem::size_of::<usize>(),
            )
        };

        let _ = file.read(&mut raw)?;

        mem::forget(raw);

        let mut auxv = AuxVec {
            hwcap: None,
            hwcap2: None,
        };

        for el in buf.chunks(2) {
            if el[0] == AT::HWCAP as usize {
                auxv.hwcap = Some(el[1]);
            }
            if el[0] == AT::HWCAP2 as usize {
                auxv.hwcap2 = Some(el[1]);
            }
        }

        Ok(auxv)
    }

    /// Returns the value for the AT key
    pub fn lookup(&self, key: AT) -> Option<usize> {
        match key {
            AT::HWCAP => self.hwcap,
            AT::HWCAP2 => self.hwcap2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn test_auxvec_linux() {
        let auxvec = AuxVec::new().unwrap();
        println!("{:?}", auxvec.lookup(AT::HWCAP));
        println!("{:?}", auxvec);
    }
}
