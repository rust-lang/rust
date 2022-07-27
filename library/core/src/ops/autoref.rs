#![allow(missing_docs, missing_debug_implementations)]
// Poor man's `k#autoref` operator.
//
// See https://github.com/rust-lang/rust/issues/99684
// and https://rust-lang.zulipchat.com/#narrow/stream/213817-t-lang/topic/Desired.20behavior.20of.20.60write!.60.20is.20unimplementable
// for some more context about this idea.
//
// Right now we polyfill this idea, by reducing support to `&mut`-autoref-only
// (effectively "just" preventing an unnecessary `&mut` level of indirection
// from being applied for a thing already behind a `&mut …`), which happens to
// work for `&mut`-based `.write_fmt()` methods, and some cases of `&`-based
// `.write_fmt()` —the whole duck-typed design / API of `write!` is asking for
// trouble—, but won't work for a `self`-based `.write_fmt()`, as pointed out
// here: https://github.com/rust-lang/rust/pull/100202#pullrequestreview-1064499226
//
// Finally, in order to reduce the chances of name conflicts as much as
// possible, the method name is a bit mangled, and to prevent usage of this
// method in stable-rust, an unstable const generic parameter that needs to be
// turbofished is added to it as well.

/// The unstable const generic parameter achieving the "unstable seal" effect.
#[unstable(feature = "autoref", issue = "none")]
#[derive(Eq, PartialEq)]
pub struct UnstableMethodSeal;

#[unstable(feature = "autoref", issue = "none")]
pub trait AutoRef {
    #[unstable(feature = "autoref", issue = "none")]
    #[inline(always)]
    fn __rustc_unstable_auto_ref_mut_helper<const _SEAL: UnstableMethodSeal>(
        &mut self,
    ) -> &mut Self {
        self
    }
}

#[unstable(feature = "autoref", issue = "none")]
impl<T: ?Sized> AutoRef for T {}

#[unstable(feature = "autoref", issue = "none")]
#[allow_internal_unstable(autoref)]
#[rustc_macro_transparency = "semitransparent"]
pub macro autoref_mut($x:expr) {{
    use $crate::ops::autoref::AutoRef as _;
    $x.__rustc_unstable_auto_ref_mut_helper::<{ $crate::ops::autoref::UnstableMethodSeal }>()
}}
