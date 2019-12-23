//! FIXME: write short doc here

use std::fmt;

use crate::db::HirDatabase;

pub struct HirFormatter<'a, 'b, DB> {
    pub db: &'a DB,
    fmt: &'a mut fmt::Formatter<'b>,
    buf: String,
    curr_size: usize,
    max_size: Option<usize>,
    should_display_default_types: bool,
}

pub trait HirDisplay {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result;

    fn display<'a, DB>(&'a self, db: &'a DB) -> HirDisplayWrapper<'a, DB, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper(db, self, None, true)
    }

    fn display_truncated<'a, DB>(
        &'a self,
        db: &'a DB,
        max_size: Option<usize>,
    ) -> HirDisplayWrapper<'a, DB, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper(db, self, max_size, false)
    }
}

impl<'a, 'b, DB> HirFormatter<'a, 'b, DB>
where
    DB: HirDatabase,
{
    pub fn write_joined<T: HirDisplay>(
        &mut self,
        iter: impl IntoIterator<Item = T>,
        sep: &str,
    ) -> fmt::Result {
        let mut first = true;
        for e in iter {
            if !first {
                write!(self, "{}", sep)?;
            }
            first = false;
            e.hir_fmt(self)?;
        }
        Ok(())
    }

    /// This allows using the `write!` macro directly with a `HirFormatter`.
    pub fn write_fmt(&mut self, args: fmt::Arguments) -> fmt::Result {
        // We write to a buffer first to track output size
        self.buf.clear();
        fmt::write(&mut self.buf, args)?;
        self.curr_size += self.buf.len();

        // Then we write to the internal formatter from the buffer
        self.fmt.write_str(&self.buf)
    }

    pub fn should_truncate(&self) -> bool {
        if let Some(max_size) = self.max_size {
            self.curr_size >= max_size
        } else {
            false
        }
    }

    pub fn should_display_default_types(&self) -> bool {
        self.should_display_default_types
    }
}

pub struct HirDisplayWrapper<'a, DB, T>(&'a DB, &'a T, Option<usize>, bool);

impl<'a, DB, T> fmt::Display for HirDisplayWrapper<'a, DB, T>
where
    DB: HirDatabase,
    T: HirDisplay,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.hir_fmt(&mut HirFormatter {
            db: self.0,
            fmt: f,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: self.2,
            should_display_default_types: self.3,
        })
    }
}
