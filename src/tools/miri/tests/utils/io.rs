use core::fmt::{self, Write};

use super::miri_extern;

pub struct MiriStderr;

impl Write for MiriStderr {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        unsafe {
            miri_extern::miri_write_to_stderr(s.as_bytes());
        }
        Ok(())
    }
}

pub struct MiriStdout;

impl Write for MiriStdout {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        unsafe {
            miri_extern::miri_write_to_stdout(s.as_bytes());
        }
        Ok(())
    }
}
