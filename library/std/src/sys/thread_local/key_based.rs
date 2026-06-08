use hashbrown::HashTable;
use hashbrown::hash_table::Entry;

use super::abort_on_dtor_unwind;
use super::key::{LazyKey, get, set};
use crate::alloc::{Allocator, Layout, System, handle_alloc_error};
use crate::cell::{Cell, RefCell};
use crate::marker::PhantomData;
use crate::mem::{forget, needs_drop, replace, transmute};
use crate::ptr::{self, NonNull, drop_in_place};
use crate::rt::thread_cleanup;

#[doc(hidden)]
#[allow_internal_unstable(thread_local_internals)]
#[allow_internal_unsafe]
#[unstable(feature = "thread_local_internals", issue = "none")]
#[rustc_macro_transparency = "semiopaque"]
pub macro thread_local_inner {
    // NOTE: we cannot import `Key` or `LocalKey` with a `use` because that can shadow user
    // provided type or type alias with a matching name. Please update the shadowing test in
    // `tests/thread.rs` if these types are renamed.

    // used to generate the `LocalKey` value for `thread_local!`.
    (@key $t:ty, $($(#[$($align_attr:tt)*])+)?, $init:expr) => {{
        #[inline]
        fn __rust_std_internal_init_fn() -> $t { $init }

        unsafe {
            $crate::thread::LocalKey::new(|__rust_std_internal_init| {
                static __RUST_STD_INTERNAL_VAL: $crate::thread::local_impl::Key<$t>
                    = $crate::thread::local_impl::Key::new({
                        $({
                            // Ensure that attributes have valid syntax
                            // and that the proper feature gate is enabled
                            $(#[$($align_attr)*])+
                            #[allow(unused)]
                            static DUMMY: () = ();
                        })?

                        #[allow(unused_mut)]
                        let mut final_align = 1;
                        $($($crate::thread::local_impl::thread_local_inner!(@align final_align, $($align_attr)*);)+)?
                        final_align
                    });

                __RUST_STD_INTERNAL_VAL.get(__rust_std_internal_init, __rust_std_internal_init_fn)
            })
        }
    }},

    // process a single `rustc_align_static` attribute
    (@align $final_align:ident, rustc_align_static($($align:tt)*) $(, $($attr_rest:tt)+)?) => {
        let new_align: $crate::primitive::usize = $($align)*;
        if new_align > $final_align {
            $final_align = new_align;
        }

        $($crate::thread::local_impl::thread_local_inner!(@align $final_align, $($attr_rest)+);)?
    },

    // process a single `cfg_attr` attribute
    // by translating it into a `cfg`ed block and recursing.
    // https://doc.rust-lang.org/reference/conditional-compilation.html#railroad-ConfigurationPredicate

    (@align $final_align:ident, cfg_attr($cfg_pred:expr, $($cfg_rhs:tt)*) $(, $($attr_rest:tt)+)?) => {
        #[cfg($cfg_pred)]
        {
            $crate::thread::local_impl::thread_local_inner!(@align $final_align, $($cfg_rhs)*);
        }

        $($crate::thread::local_impl::thread_local_inner!(@align $final_align, $($attr_rest)+);)?
    },
}

// TLS keys are a *very* limited resource. Thus, to conserve them, this TLS
// implementation stores all variables in a per-thread map, with the key being
// the address of a `static` variable (these are unique even for immutable values).
// The map holds pointers to the individual variable's storage, as well as the
// metadata describing its memory layout.
//
// Additionally, we keep a list of the value destructors for types that need
// them and run them before deallocating the storage of values that do not need
// destruction, which matches the behaviour of the native TLS implementation.

#[expect(missing_debug_implementations)]
pub struct Key<T> {
    // `Key` must not be zero-sized, since it has to have a unique address.
    // Thus we can conserve space by also storing the requested alignment here
    // (we store the power-of-two instead of the actual alignment to keep `Key`
    // as small as possible).
    align: u8,
    _ty: PhantomData<Cell<T>>,
}

impl<T> Key<T> {
    pub const fn new(align: usize) -> Key<T> {
        let Ok(layout) = Layout::new::<T>().align_to(align) else {
            panic!("invalid alignment for TLS variable");
        };
        Key { align: layout.align().ilog2() as u8, _ty: PhantomData }
    }

    pub fn get(&'static self, i: Option<&mut Option<T>>, f: impl FnOnce() -> T) -> *const T {
        let key = <*const u8>::addr(&self.align);
        if let Some(ptr) = find(key) {
            // Either the value was initialized previously or has already been
            // destroyed. Both scenarios are handled by the consumer of the
            // pointer.
            ptr.map_or(ptr::null_mut(), |p| p.as_ptr()).cast()
        } else {
            // The value has not been initialized on this thread yet.

            let val = i.and_then(Option::take).unwrap_or_else(f);

            // SAFETY: we ensure that `self.align` is sufficient for `T` in `new`.
            let value = unsafe { Value::new(&self.align, val) };
            let dtor = if needs_drop::<T>() {
                // SAFETY: thin pointers are ABI compatible regardless of their type.
                Some(unsafe {
                    transmute::<unsafe fn(*mut T), unsafe fn(*mut u8)>(drop_in_place::<T>)
                })
            } else {
                None
            };

            // Note that if `f` recursively initialized the variable, this will
            // overwrite the existing value. We might want to change that in the
            // future, but for now let's preserve that (sound) behaviour.
            // (c.f. https://github.com/rust-lang/rust/issues/110897#issuecomment-1525705682
            // and the ensuing discussion)
            unsafe { insert(value, dtor).map_or(ptr::null_mut(), |p| p.as_ptr()).cast() }
        }
    }
}

unsafe impl<T> Send for Key<T> {}
unsafe impl<T> Sync for Key<T> {}

struct Value {
    key_align: &'static u8,
    size: usize,
    // A pointer to the value. If there is none, it means the value has been
    // destroyed already.
    ptr: Option<NonNull<u8>>,
}

impl Value {
    /// Creates a new `Value` by allocating storage for `T` with alignment
    /// matching the requested `key_align` and storing `val` in it.
    ///
    /// # Safety
    /// `1 << key_align` must be higher than the minimum alignment of `T`.
    unsafe fn new<T>(key_align: &'static u8, val: T) -> Value {
        debug_assert!((1 << key_align) >= align_of::<T>());
        let mut value = Value { key_align, size: size_of::<T>(), ptr: None };

        let layout = value.layout();
        value.ptr = if layout.size() != 0 {
            let Ok(ptr) = System.allocate(layout) else { handle_alloc_error(layout) };

            // SAFETY: `ptr` points to valid memory with a layout sufficient
            // to hold `T`.
            unsafe { ptr.cast::<T>().write(val) };
            Some(ptr.cast())
        } else {
            // Forge a pointer with the requested alignment. Since `T` is zero-
            // sized, there is nothing to store. But do avoid dropping `val`.
            forget(val);
            Some(layout.dangling_ptr())
        };

        value
    }

    fn layout(&self) -> Layout {
        let align = 1 << self.key_align;
        Layout::from_size_align(self.size, align).unwrap()
    }

    fn key(&self) -> usize {
        <*const u8>::addr(self.key_align)
    }

    fn hash(&self) -> u64 {
        // Its quite likely that all `Key`s are laid out right next to each
        // other, in which case the lowest bits will have the highest variance.
        // Since `HashTable` uses the lowest bits for its bucked index that
        // makes the key good enough as a hash. If this turns out to result in
        // too many conflicts, one might try incorporating the the page number
        // into the top 7 bits (which are used in the hash table control bytes).
        self.key() as u64
    }
}

static STORAGE_KEY: LazyKey = LazyKey::new(Some(cleanup));

struct Storage {
    vals: HashTable<Value, System>,
    dtors: Vec<(usize, unsafe fn(*mut u8)), System>,
}

/// Preinitialize the thread-local state.
///
/// This also ensures that `thread_cleanup` is run, even if no other variables
/// are initialized.
#[cfg(not(any(target_os = "windows", target_os = "xous")))]
pub(crate) fn init() {
    let key = STORAGE_KEY.force();
    if unsafe { get(key).is_null() } {
        let (storage, _) = Box::into_raw_with_allocator(Box::new_in(
            RefCell::new(Storage { vals: HashTable::new_in(System), dtors: Vec::new_in(System) }),
            System,
        ));
        unsafe { set(key, storage.cast()) };
    }
}

/// Finds the pointer associated with the specified key, if it has been
/// initialized already.
fn find(key: usize) -> Option<Option<NonNull<u8>>> {
    let storage_key = STORAGE_KEY.force();
    // Retrieve the thread-local storage. If it has not been initialized yet,
    // the value can't have been initialized either – except if thread-local
    // storage has previously been destroyed. But its impossible to detect that
    // without causing other issues (such as previously-uninitialized variables
    // not being available in foreign thread-local destructors). This is
    // irrelevant for nearly all users anyway, since there is no way in `std` to
    // run code after the TLS destructors.
    let storage = unsafe { get(storage_key).cast::<RefCell<Storage>>().as_ref()? };
    storage.borrow().vals.find(key as u64, |v| v.key() == key).map(|v| v.ptr)
}

/// Inserts `value` and its corresponding destructor, potentially replacing a
/// previous value (and destroying it with `dtor` if applicable).
unsafe fn insert(value: Value, dtor: Option<unsafe fn(*mut u8)>) -> Option<NonNull<u8>> {
    let ptr = value.ptr;

    let storage_key = STORAGE_KEY.force();
    let storage = unsafe { get(storage_key).cast::<RefCell<Storage>>().as_ref() };
    if let Some(storage) = storage {
        let mut storage = storage.borrow_mut();
        match storage.vals.entry(value.hash(), |v| v.key() == value.key(), Value::hash) {
            Entry::Occupied(mut existing) => {
                let existing = existing.get_mut();
                debug_assert_eq!(existing.size, value.size);

                if let Some(old) = replace(&mut existing.ptr, ptr) {
                    let layout = existing.layout();
                    // There was a value already in place. Overwrite it.

                    // Release the borrow to allow access to other TLS variables
                    // in the destructor.
                    drop(storage);

                    if let Some(dtor) = dtor {
                        unsafe { dtor(old.as_ptr()) };
                    }

                    if layout.size() != 0 {
                        unsafe { System.deallocate(old, layout) };
                    }
                }
            }
            Entry::Vacant(vacancy) => {
                let key = value.key();
                vacancy.insert(value);
                if let Some(dtor) = dtor {
                    storage.dtors.push((key, dtor));
                }
            }
        }
    } else {
        // If the thread-local storage has not been allocated yet, initialize it
        // and take the opportunity to pre-insert the value before wrapping the
        // storage in a `RefCell`.
        let mut dtors = Vec::new_in(System);
        if let Some(dtor) = dtor {
            dtors.push((value.key(), dtor));
        }

        let mut vals = HashTable::new_in(System);
        vals.insert_unique(value.hash(), value, Value::hash);

        let (storage, _) = Box::into_raw_with_allocator(Box::new_in(
            RefCell::new(Storage { vals, dtors }),
            System,
        ));
        unsafe { set(storage_key, storage.cast()) };
    }

    // Return the pointer to allow this function to be tail-called from `get`.
    ptr
}

unsafe extern "C" fn cleanup(storage: *mut u8) {
    let storage_key = STORAGE_KEY.force();
    // Restore the pointer to the thread-local state so that destructors may
    // access other TLS variables.
    unsafe { set(storage_key, storage) };

    let storage = storage as *mut RefCell<Storage>;
    let store = unsafe { &*(storage as *const RefCell<Storage>) };
    let mut s = store.borrow_mut();
    while let Some((key, dtor)) = s.dtors.pop() {
        let value = s.vals.find_mut(key as u64, |v| v.key() == key);
        if let Some(value) = value {
            // Mark the value as destroyed.
            if let Some(ptr) = value.ptr.take() {
                let layout = value.layout();
                // Release the borrow to allow access to TLS variables in the
                // destructor.
                drop(s);

                abort_on_dtor_unwind(|| unsafe { dtor(ptr.as_ptr()) });

                if layout.size() != 0 {
                    unsafe { System.deallocate(ptr, layout) };
                }

                s = store.borrow_mut();
            }
        }
    }

    drop(s);
    thread_cleanup();

    // Unregister the thread-local state.
    unsafe { set(storage_key, ptr::null_mut()) };
    let mut storage = unsafe { Box::from_raw_in(storage, System) };

    // Deallocate the storage of all remaining variables.
    for value in &mut storage.get_mut().vals {
        if let Some(ptr) = value.ptr
            && value.size != 0
        {
            unsafe { System.deallocate(ptr, value.layout()) };
        }
    }
}

#[rustc_macro_transparency = "semiopaque"]
pub(crate) macro local_pointer {
    () => {},
    ($vis:vis static $name:ident; $($rest:tt)*) => {
        $vis static $name: $crate::sys::thread_local::LocalPointer = $crate::sys::thread_local::LocalPointer::__new();
        $crate::sys::thread_local::local_pointer! { $($rest)* }
    },
}

pub(crate) struct LocalPointer {
    key: LazyKey,
}

impl LocalPointer {
    pub const fn __new() -> LocalPointer {
        LocalPointer { key: LazyKey::new(None) }
    }

    pub fn get(&'static self) -> *mut () {
        unsafe { get(self.key.force()) as *mut () }
    }

    pub fn set(&'static self, p: *mut ()) {
        unsafe { set(self.key.force(), p as *mut u8) }
    }
}
