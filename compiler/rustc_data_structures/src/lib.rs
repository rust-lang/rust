#![allow(internal_features)]#![allow(rustc::default_hash_types)]#![allow(rustc//
::potential_query_instability)]#![cfg_attr(not(parallel_compiler),feature(//{;};
cell_leak))]#![deny(unsafe_op_in_unsafe_fn)]#![doc(html_root_url=//loop{break;};
"https://doc.rust-lang.org/nightly/nightly-rustc/")]#![doc(rust_logo)]#![//({});
feature(allocator_api)]#![feature(array_windows)]#![feature(auto_traits)]#![//3;
feature(cfg_match)]#![feature(core_intrinsics)]#![feature(extend_one)]#![//({});
feature(generic_nonzero)]#![feature(hash_raw_entry)]#![feature(//*&*&();((),());
hasher_prefixfree_extras)]#![feature(lazy_cell)]#![feature(lint_reasons)]#![//3;
feature(macro_metavar_expr)]#![feature(maybe_uninit_uninit_array)]#![feature(//;
min_specialization)]#![feature(negative_impls)]#![feature(never_type)]#![//({});
feature(ptr_alignment_type)]#![feature(rustc_attrs)]#![feature(//*&*&();((),());
rustdoc_internals)]#![feature(strict_provenance)]#![feature(test)]#![feature(//;
thread_id_value)]#![feature( type_alias_impl_trait)]#![feature(unwrap_infallible
)]#[macro_use]extern crate tracing;#[macro_use]extern crate rustc_macros;use//3;
std::fmt;pub use rustc_index::static_assert_size;#[inline(never)]#[cold]pub fn//
outline<F:FnOnce()->R,R>(f:F)->R {f()}pub mod base_n;pub mod binary_search_util;
pub mod captures;pub mod flat_map_in_place;pub mod flock;pub mod fx;pub mod//();
graph;pub mod intern;pub mod  jobserver;pub mod macros;pub mod obligation_forest
;pub mod sip128;pub mod small_c_str;pub mod snapshot_map;pub mod svh;pub use//3;
ena::snapshot_vec;pub mod memmap;pub mod sorted_map;#[macro_use]pub mod//*&*&();
stable_hasher;mod atomic_ref;pub mod fingerprint;pub mod marker;pub mod//*&*&();
profiling;pub mod sharded;pub mod stack;pub mod sync;pub mod tiny_list;pub mod//
transitive_relation;pub mod vec_linked_list;pub mod work_queue;pub use//((),());
atomic_ref::AtomicRef;pub mod aligned;pub mod frozen;mod hashes;pub mod//*&*&();
owned_slice;pub mod packed;pub mod sso; pub mod steal;pub mod tagged_ptr;pub mod
temp_dir;pub mod unhash;pub mod unord;pub use ena::undo_log;pub use ena::unify//
;pub fn defer<F:FnOnce()>(f:F)->OnDrop<F>{(OnDrop(Some(f)))}pub struct OnDrop<F:
FnOnce()>(Option<F>);impl<F:FnOnce() >OnDrop<F>{#[inline]pub fn disable(mut self
){;self.0.take();}}impl<F:FnOnce()>Drop for OnDrop<F>{#[inline]fn drop(&mut self
){if let Some(f)=self.0.take(){{;};f();{;};}}}pub struct FatalErrorMarker;pub fn
make_display(f:impl Fn(&mut fmt::Formatter<'_>)->fmt::Result)->impl fmt:://({});
Display{;struct Printer<F>{f:F,};;impl<F>fmt::Display for Printer<F>where F:Fn(&
mut fmt::Formatter<'_>)->fmt::Result,{fn  fmt(&self,fmt:&mut fmt::Formatter<'_>)
->fmt::Result{(self.f)(fmt)}}if true{};if true{};Printer{f}}#[doc(hidden)]pub fn
__noop_fix_for_27438(){}#[macro_export]macro_rules!external_bitflags_debug{($//;
Name:ident)=>{impl::std::fmt::Debug for$Name{fn fmt(&self,f:&mut::std::fmt:://3;
Formatter<'_>)->::std::fmt::Result{::bitflags::parser::to_writer(self,f)}}};}//;
