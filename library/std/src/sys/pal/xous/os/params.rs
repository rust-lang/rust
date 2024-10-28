/// Xous passes a pointer to the parameter block as the second argument.
/// This is used for passing flags such as environment variables. The
/// format of the argument block is:
///
/// #[repr(C)]
/// struct BlockHeader {
///     /// Magic number that identifies this block. Must be printable ASCII.
///     magic: [u8; 4],
///
///     /// The size of the data block. Does not include this header. May be 0.
///     size: u32,
///
///     /// The contents of this block. Varies depending on the block type.
///     data: [u8; 0],
/// }
///
/// There is a BlockHeader at the start that has magic `AppP`, and the data
/// that follows is the number of blocks present:
///
/// #[repr(C)]
/// struct ApplicationParameters {
///     magic: b"AppP",
///     size: 4u32,
///
///     /// The size of the entire application slice, in bytes, including all headers
///     length: u32,
///
///     /// Number of application parameters present. Must be at least 1 (this block)
///     entries: (parameter_count as u32).to_bytes_le(),
/// }
///
/// #[repr(C)]
/// struct EnvironmentBlock {
///     magic: b"EnvB",
///
///     /// Total number of bytes, excluding this header
///     size: 2+data.len(),
///
///     /// The number of environment variables
///     count: u16,
///
///     /// Environment variable iteration
///     data: [u8; 0],
/// }
///
/// Environment variables are present in an `EnvB` block. The `data` section is
/// a sequence of bytes of the form:
///
///      (u16 /* key_len */; [0u8; key_len as usize] /* key */,
///       u16 /* val_len */ [0u8; val_len as usize])
///
/// #[repr(C)]
/// struct ArgumentList {
///     magic: b"ArgL",
///
///     /// Total number of bytes, excluding this header
///     size: 2+data.len(),
///
///     /// The number of arguments variables
///     count: u16,
///
///     /// Argument variable iteration
///     data: [u8; 0],
/// }
///
/// Args are just an array of strings that represent command line arguments.
/// They are a sequence of the form:
///
///      (u16 /* val_len */ [0u8; val_len as usize])
use core::slice;

use crate::ffi::OsString;

/// Magic number indicating we have an environment block
const ENV_MAGIC: [u8; 4] = *b"EnvB";

/// Command line arguments list
const ARGS_MAGIC: [u8; 4] = *b"ArgL";

/// Magic number indicating the loader has passed application parameters
const PARAMS_MAGIC: [u8; 4] = *b"AppP";

#[cfg(test)]
mod tests;

pub(crate) struct ApplicationParameters {
    data: &'static [u8],
    offset: usize,
    _entries: usize,
}

impl ApplicationParameters {
    pub(crate) unsafe fn new_from_ptr(data: *const u8) -> Option<ApplicationParameters> {
        if data.is_null() {
            return None;
        }

        let magic = unsafe { core::slice::from_raw_parts(data, 4) };
        let block_length = unsafe {
            u32::from_le_bytes(slice::from_raw_parts(data.add(4), 4).try_into().ok()?) as usize
        };
        let data_length = unsafe {
            u32::from_le_bytes(slice::from_raw_parts(data.add(8), 4).try_into().ok()?) as usize
        };
        let entries = unsafe {
            u32::from_le_bytes(slice::from_raw_parts(data.add(12), 4).try_into().ok()?) as usize
        };

        // Check for the main header
        if data_length < 16 || magic != PARAMS_MAGIC || block_length != 8 {
            return None;
        }

        let data = unsafe { slice::from_raw_parts(data, data_length) };

        Some(ApplicationParameters { data, offset: 0, _entries: entries })
    }
}

impl Iterator for ApplicationParameters {
    type Item = ApplicationParameter;

    fn next(&mut self) -> Option<Self::Item> {
        // Fetch magic, ensuring we don't run off the end
        if self.offset + 4 > self.data.len() {
            return None;
        }
        let magic = &self.data[self.offset..self.offset + 4];
        self.offset += 4;

        // Fetch header size
        if self.offset + 4 > self.data.len() {
            return None;
        }
        let size = u32::from_le_bytes(self.data[self.offset..self.offset + 4].try_into().unwrap())
            as usize;
        self.offset += 4;

        // Fetch data contents
        if self.offset + size > self.data.len() {
            return None;
        }
        let data = &self.data[self.offset..self.offset + size];
        self.offset += size;

        Some(ApplicationParameter { data, magic: magic.try_into().unwrap() })
    }
}

pub(crate) struct ApplicationParameter {
    data: &'static [u8],
    magic: [u8; 4],
}

pub(crate) struct ApplicationParameterError;

pub(crate) struct EnvironmentBlock {
    _count: usize,
    data: &'static [u8],
    offset: usize,
}

impl TryFrom<&ApplicationParameter> for EnvironmentBlock {
    type Error = ApplicationParameterError;

    fn try_from(value: &ApplicationParameter) -> Result<Self, Self::Error> {
        if value.data.len() < 2 || value.magic != ENV_MAGIC {
            return Err(ApplicationParameterError);
        }

        let count = u16::from_le_bytes(value.data[0..2].try_into().unwrap()) as usize;

        Ok(EnvironmentBlock { data: &value.data[2..], offset: 0, _count: count })
    }
}

pub(crate) struct EnvironmentEntry {
    pub key: &'static str,
    pub value: &'static str,
}

impl Iterator for EnvironmentBlock {
    type Item = EnvironmentEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + 2 > self.data.len() {
            return None;
        }
        let key_len =
            u16::from_le_bytes(self.data[self.offset..self.offset + 2].try_into().ok()?) as usize;
        self.offset += 2;

        if self.offset + key_len > self.data.len() {
            return None;
        }
        let key = core::str::from_utf8(&self.data[self.offset..self.offset + key_len]).ok()?;
        self.offset += key_len;

        if self.offset + 2 > self.data.len() {
            return None;
        }
        let value_len =
            u16::from_le_bytes(self.data[self.offset..self.offset + 2].try_into().ok()?) as usize;
        self.offset += 2;

        if self.offset + value_len > self.data.len() {
            return None;
        }
        let value = core::str::from_utf8(&self.data[self.offset..self.offset + value_len]).ok()?;
        self.offset += value_len;

        Some(EnvironmentEntry { key, value })
    }
}

pub(crate) struct ArgumentList {
    data: &'static [u8],
    _count: usize,
    offset: usize,
}

impl TryFrom<&ApplicationParameter> for ArgumentList {
    type Error = ApplicationParameterError;

    fn try_from(value: &ApplicationParameter) -> Result<Self, Self::Error> {
        if value.data.len() < 2 || value.magic != ARGS_MAGIC {
            return Err(ApplicationParameterError);
        }
        let count =
            u16::from_le_bytes(value.data[0..2].try_into().or(Err(ApplicationParameterError))?)
                as usize;
        Ok(ArgumentList { data: &value.data[2..], _count: count, offset: 0 })
    }
}

pub(crate) struct ArgumentEntry {
    value: &'static str,
}

impl Into<&str> for ArgumentEntry {
    fn into(self) -> &'static str {
        self.value
    }
}

impl Into<OsString> for ArgumentEntry {
    fn into(self) -> OsString {
        self.value.into()
    }
}

impl Iterator for ArgumentList {
    type Item = ArgumentEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + 2 > self.data.len() {
            return None;
        }
        let value_len =
            u16::from_le_bytes(self.data[self.offset..self.offset + 2].try_into().ok()?) as usize;
        self.offset += 2;

        if self.offset + value_len > self.data.len() {
            return None;
        }
        let value = core::str::from_utf8(&self.data[self.offset..self.offset + value_len]).ok()?;
        self.offset += value_len;

        Some(ArgumentEntry { value })
    }
}
