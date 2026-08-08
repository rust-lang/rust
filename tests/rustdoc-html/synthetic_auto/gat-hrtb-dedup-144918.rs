// Regression test for <https://github.com/rust-lang/rust/issues/144918>.
//
// Synthesizing auto trait impls used to ICE ("Unable to fulfill trait ...:
// [Ambiguity]") when a manual `unsafe impl` carried a higher-ranked bound on a
// generic associated type (`for<'w> A::Wired<'w>: Send`) while a field of the
// type being documented mentioned the same GAT with a concrete lifetime
// (`A::Wired<'static>`). The two discovered bounds differ only in a region
// *inside* a type argument, which `add_user_clause` failed to deduplicate,
// leaving an ambiguous `ParamEnv`.

pub trait Alloc {
    type Wired<'w>;
}

pub struct Slot<A>(A);

unsafe impl<A: Alloc> Send for Slot<A> where for<'w> A::Wired<'w>: Send {}

//@ has gat_hrtb_dedup_144918/struct.Custody.html
//
// The two discovered `Send` bounds collapse into the stricter, higher-ranked
// one.
// FIXME: `for<'w>` is rendered on the bound rather than before the self type;
// this is a pre-existing rendering quirk.
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<A> Send for Custody<A>where <A as Alloc>::Wired<'w>: for<'w> Send,"
//
// `Sync` discovers no higher-ranked bound, so the concrete one is kept.
//@ has - '//*[@id="synthetic-implementations-list"]//*[@class="impl"]//h3[@class="code-header"]' \
// "impl<A> Sync for Custody<A>where <A as Alloc>::Wired<'static>: Sync, A: Sync,"
pub struct Custody<A: Alloc>(Slot<A>, A::Wired<'static>);
