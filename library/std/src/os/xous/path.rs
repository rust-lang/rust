#![stable(feature = "rust1", since = "1.0.0")]

use crate::path;

#[stable(feature = "file_type_ext", since = "1.5.0")]
pub trait PathExt {
    /// Returns `true` if this file type is a basis.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::fs;
    /// use std::os::xous::fs::FileTypeExt;
    /// use std::io;
    ///
    /// fn main() -> io::Result<()> {
    ///     let meta = fs::metadata(":.System")?;
    ///     let file_type = meta.file_type();
    ///     assert!(file_type.is_basis());
    ///     Ok(())
    /// }
    /// ```
    #[stable(feature = "file_type_ext", since = "1.5.0")]
    fn is_basis(&self) -> bool;
}

#[stable(feature = "file_type_ext", since = "1.5.0")]
impl PathExt for path::Path {
    fn is_basis(&self) -> bool {
        let Some(as_str) = self.as_os_str().to_str() else {
            return false;
        };

        let Ok(path) = crate::sys::path::split_basis_dict_key(as_str) else {
            return false;
        };

        path.basis.is_some() && path.dict.is_none() && path.key.is_none()
    }
}
