#[rustc_on_unimplemented(message = "`{Self}` doesn't implement `DynSend`. \
            Add it to `rustc_data_structures::marker` or use `IntoDynSyncSend` if it's already `Send`")]
// This is an auto trait for types which can be sent across threads if `sync::is_dyn_thread_safe()`
// is true. These types can be wrapped in a `FromDyn` to get a `Send` type. Wrapping a
// `Send` type in `IntoDynSyncSend` will create a `DynSend` type.
pub unsafe auto trait DynSend {}

#[rustc_on_unimplemented(message = "`{Self}` doesn't implement `DynSync`. \
            Add it to `rustc_data_structures::marker` or use `IntoDynSyncSend` if it's already `Sync`")]
// This is an auto trait for types which can be shared across threads if `sync::is_dyn_thread_safe()`
// is true. These types can be wrapped in a `FromDyn` to get a `Sync` type. Wrapping a
// `Sync` type in `IntoDynSyncSend` will create a `DynSync` type.
pub unsafe auto trait DynSync {}

// Same with `Sync` and `Send`.
unsafe impl<T: DynSync + ?Sized> DynSend for &T {}

macro_rules! impls_dyn_send_neg {
    ($([$t1: ty $(where $($generics1: tt)*)?])*) => {
        $(impl$(<$($generics1)*>)? !DynSend for $t1 {})*
    };
}

// Consistent with `std`
impls_dyn_send_neg!(
    [std::env::Args]
    [std::env::ArgsOs]
    [*const T where T: ?Sized]
    [*mut T where T: ?Sized]
    [std::ptr::NonNull<T> where T: ?Sized]
    [std::rc::Rc<T> where T: ?Sized]
    [std::rc::Weak<T> where T: ?Sized]
    [std::sync::MutexGuard<'_, T> where T: ?Sized]
    [std::sync::RwLockReadGuard<'_, T> where T: ?Sized]
    [std::sync::RwLockWriteGuard<'_, T> where T: ?Sized]
    [std::io::StdoutLock<'_>]
    [std::io::StderrLock<'_>]
);

#[cfg(any(unix, target_os = "hermit", target_os = "wasi", target_os = "solid_asp3"))]
// Consistent with `std`, `os_imp::Env` is `!Sync` in these platforms
impl !DynSend for std::env::VarsOs {}

macro_rules! already_send {
    ($([$ty: ty])*) => {
        $(unsafe impl DynSend for $ty where $ty: Send {})*
    };
}

// These structures are already `Send`.
already_send!(
    [std::backtrace::Backtrace][std::io::Stdout][std::io::Stderr][std::io::Error][std::fs::File]
        [rustc_arena::DroplessArena][crate::memmap::Mmap][crate::profiling::SelfProfiler]
        [crate::owned_slice::OwnedSlice]
);

macro_rules! impl_dyn_send {
    ($($($attr: meta)* [$ty: ty where $($generics2: tt)*])*) => {
        $(unsafe impl<$($generics2)*> DynSend for $ty {})*
    };
}

impl_dyn_send!(
    [std::sync::atomic::AtomicPtr<T> where T]
    [std::sync::Mutex<T> where T: ?Sized+ DynSend]
    [std::sync::mpsc::Sender<T> where T: DynSend]
    [std::sync::Arc<T> where T: ?Sized + DynSync + DynSend]
    [std::sync::LazyLock<T, F> where T: DynSend, F: DynSend]
    [std::collections::HashSet<K, S> where K: DynSend, S: DynSend]
    [std::collections::HashMap<K, V, S> where K: DynSend, V: DynSend, S: DynSend]
    [std::collections::BTreeMap<K, V, A> where K: DynSend, V: DynSend, A: std::alloc::Allocator + Clone + DynSend]
    [Vec<T, A> where T: DynSend, A: std::alloc::Allocator + DynSend]
    [Box<T, A> where T: ?Sized + DynSend, A: std::alloc::Allocator + DynSend]
    [crate::sync::RwLock<T> where T: DynSend]
    [crate::tagged_ptr::CopyTaggedPtr<P, T, CP> where P: Send + crate::tagged_ptr::Pointer, T: Send + crate::tagged_ptr::Tag, const CP: bool]
    [rustc_arena::TypedArena<T> where T: DynSend]
    [indexmap::IndexSet<V, S> where V: DynSend, S: DynSend]
    [indexmap::IndexMap<K, V, S> where K: DynSend, V: DynSend, S: DynSend]
    [thin_vec::ThinVec<T> where T: DynSend]
    [smallvec::SmallVec<A> where A: smallvec::Array + DynSend]
);

macro_rules! impls_dyn_sync_neg {
    ($([$t1: ty $(where $($generics1: tt)*)?])*) => {
        $(impl$(<$($generics1)*>)? !DynSync for $t1 {})*
    };
}

// Consistent with `std`
impls_dyn_sync_neg!(
    [std::env::Args]
    [std::env::ArgsOs]
    [*const T where T: ?Sized]
    [*mut T where T: ?Sized]
    [std::cell::Cell<T> where T: ?Sized]
    [std::cell::RefCell<T> where T: ?Sized]
    [std::cell::UnsafeCell<T> where T: ?Sized]
    [std::ptr::NonNull<T> where T: ?Sized]
    [std::rc::Rc<T> where T: ?Sized]
    [std::rc::Weak<T> where T: ?Sized]
    [std::cell::OnceCell<T> where T]
    [std::sync::mpsc::Receiver<T> where T]
    [std::sync::mpsc::Sender<T> where T]
);

#[cfg(any(unix, target_os = "hermit", target_os = "wasi", target_os = "solid_asp3"))]
// Consistent with `std`, `os_imp::Env` is `!Sync` in these platforms
impl !DynSync for std::env::VarsOs {}

macro_rules! already_sync {
    ($([$ty: ty])*) => {
        $(unsafe impl DynSync for $ty where $ty: Sync {})*
    };
}

// These structures are already `Sync`.
already_sync!(
    [std::sync::atomic::AtomicBool][std::sync::atomic::AtomicUsize][std::sync::atomic::AtomicU8]
        [std::sync::atomic::AtomicU32][std::backtrace::Backtrace][std::io::Error][std::fs::File]
        [jobserver_crate::Client][crate::memmap::Mmap][crate::profiling::SelfProfiler]
        [crate::owned_slice::OwnedSlice]
);

// Use portable AtomicU64 for targets without native 64-bit atomics
#[cfg(target_has_atomic = "64")]
already_sync!([std::sync::atomic::AtomicU64]);

#[cfg(not(target_has_atomic = "64"))]
already_sync!([portable_atomic::AtomicU64]);

macro_rules! impl_dyn_sync {
    ($($($attr: meta)* [$ty: ty where $($generics2: tt)*])*) => {
        $(unsafe impl<$($generics2)*> DynSync for $ty {})*
    };
}

impl_dyn_sync!(
    [std::sync::atomic::AtomicPtr<T> where T]
    [std::sync::OnceLock<T> where T: DynSend + DynSync]
    [std::sync::Mutex<T> where T: ?Sized + DynSend]
    [std::sync::Arc<T> where T: ?Sized + DynSync + DynSend]
    [std::sync::LazyLock<T, F> where T: DynSend + DynSync, F: DynSend]
    [std::collections::HashSet<K, S> where K: DynSync, S: DynSync]
    [std::collections::HashMap<K, V, S> where K: DynSync, V: DynSync, S: DynSync]
    [std::collections::BTreeMap<K, V, A> where K: DynSync, V: DynSync, A: std::alloc::Allocator + Clone + DynSync]
    [Vec<T, A> where T: DynSync, A: std::alloc::Allocator + DynSync]
    [Box<T, A> where T: ?Sized + DynSync, A: std::alloc::Allocator + DynSync]
    [crate::sync::RwLock<T> where T: DynSend + DynSync]
    [crate::sync::WorkerLocal<T> where T: DynSend]
    [crate::intern::Interned<'a, T> where 'a, T: DynSync]
    [crate::tagged_ptr::CopyTaggedPtr<P, T, CP> where P: Sync + crate::tagged_ptr::Pointer, T: Sync + crate::tagged_ptr::Tag, const CP: bool]
    [parking_lot::lock_api::Mutex<R, T> where R: DynSync, T: ?Sized + DynSend]
    [parking_lot::lock_api::RwLock<R, T> where R: DynSync, T: ?Sized + DynSend + DynSync]
    [indexmap::IndexSet<V, S> where V: DynSync, S: DynSync]
    [indexmap::IndexMap<K, V, S> where K: DynSync, V: DynSync, S: DynSync]
    [smallvec::SmallVec<A> where A: smallvec::Array + DynSync]
    [thin_vec::ThinVec<T> where T: DynSync]
);

pub fn assert_dyn_sync<T: ?Sized + DynSync>() {}
pub fn assert_dyn_send<T: ?Sized + DynSend>() {}
pub fn assert_dyn_send_val<T: ?Sized + DynSend>(_t: &T) {}
pub fn assert_dyn_send_sync_val<T: ?Sized + DynSync + DynSend>(_t: &T) {}

#[derive(Copy, Clone)]
pub struct FromDyn<T>(T);

impl<T> FromDyn<T> {
    #[inline(always)]
    pub fn from(val: T) -> Self {
        // Check that `sync::is_dyn_thread_safe()` is true on creation so we can
        // implement `Send` and `Sync` for this structure when `T`
        // implements `DynSend` and `DynSync` respectively.
        assert!(crate::sync::is_dyn_thread_safe());
        FromDyn(val)
    }

    #[inline(always)]
    pub fn into_inner(self) -> T {
        self.0
    }
}

// `FromDyn` is `Send` if `T` is `DynSend`, since it ensures that sync::is_dyn_thread_safe() is true.
unsafe impl<T: DynSend> Send for FromDyn<T> {}

// `FromDyn` is `Sync` if `T` is `DynSync`, since it ensures that sync::is_dyn_thread_safe() is true.
unsafe impl<T: DynSync> Sync for FromDyn<T> {}

impl<T> std::ops::Deref for FromDyn<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// A wrapper to convert a struct that is already a `Send` or `Sync` into
// an instance of `DynSend` and `DynSync`, since the compiler cannot infer
// it automatically in some cases. (e.g. Box<dyn Send / Sync>)
#[derive(Copy, Clone)]
pub struct IntoDynSyncSend<T: ?Sized>(pub T);

unsafe impl<T: ?Sized + Send> DynSend for IntoDynSyncSend<T> {}
unsafe impl<T: ?Sized + Sync> DynSync for IntoDynSyncSend<T> {}

impl<T> std::ops::Deref for IntoDynSyncSend<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> std::ops::DerefMut for IntoDynSyncSend<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}
