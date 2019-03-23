//! In certain situations, rust automatically inserts derefs as necessary: for
//! example, field accesses `foo.bar` still work when `foo` is actually a
//! reference to a type with the field `bar`. This is an approximation of the
//! logic in rustc (which lives in librustc_typeck/check/autoderef.rs).

use ra_syntax::algo::generate;

use crate::HirDatabase;
use super::Ty;

impl Ty {
    /// Iterates over the possible derefs of `ty`.
    pub fn autoderef<'a>(self, db: &'a impl HirDatabase) -> impl Iterator<Item = Ty> + 'a {
        generate(Some(self), move |ty| ty.autoderef_step(db))
    }

    fn autoderef_step(&self, _db: &impl HirDatabase) -> Option<Ty> {
        // FIXME Deref::deref
        self.builtin_deref()
    }
}
