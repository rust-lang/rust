pub enum CopyState {
    #[cfg_attr(not(any(target_os = "linux", target_os = "android")), expect(dead_code))]
    Ended(u64),
    Fallback(u64),
}

cfg_select! {
    any(target_os = "linux", target_os = "android") => {
        mod linux;
        pub use linux::kernel_copy;
    }
    _ => {
        use crate::io::{Result, Read, Write};

        pub fn kernel_copy<R: ?Sized, W: ?Sized>(_reader: &mut R, _writer: &mut W) -> Result<CopyState>
        where
            R: Read,
            W: Write,
        {
            Ok(CopyState::Fallback(0))
        }
    }
}
