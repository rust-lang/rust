use std::fmt;

use crate::db::HirDatabase;

pub struct HirFormatter<'a, 'b, DB> {
    pub db: &'a DB,
    fmt: &'a mut fmt::Formatter<'b>,
}

pub trait HirDisplay {
    fn hir_fmt(&self, f: &mut HirFormatter<impl HirDatabase>) -> fmt::Result;
    fn display<'a, DB>(&'a self, db: &'a DB) -> HirDisplayWrapper<'a, DB, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper(db, self)
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
        fmt::write(self.fmt, args)
    }
}

pub struct HirDisplayWrapper<'a, DB, T>(&'a DB, &'a T);

impl<'a, DB, T> fmt::Display for HirDisplayWrapper<'a, DB, T>
where
    DB: HirDatabase,
    T: HirDisplay,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.hir_fmt(&mut HirFormatter { db: self.0, fmt: f })
    }
}
