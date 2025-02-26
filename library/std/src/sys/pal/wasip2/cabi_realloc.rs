//! This module contains a canonical definition of the `cabi_realloc` function
//! for the component model.
//!
//! The component model's canonical ABI for representing datatypes in memory
//! makes use of this function when transferring lists and strings, for example.
//! This function behaves like C's `realloc` but also takes alignment into
//! account.
//!
//! Components are notably not required to export this function, but nearly
//! all components end up doing so currently. This definition in the standard
//! library removes the need for all compilations to define this themselves.
//!
//! More information about the canonical ABI can be found at
//! <https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md>
//!
//! Note that the name of this function is not standardized in the canonical ABI
//! at this time. Instead it's a convention of the "componentization process"
//! where a core wasm module is converted to a component to use this name.
//! Additionally this is not the only possible definition of this function, so
//! this is defined as a "weak" symbol. This means that other definitions are
//! allowed to overwrite it if they are present in a compilation.

use crate::alloc::{self, Layout};
use crate::ptr;

#[used]
static FORCE_CODEGEN_OF_CABI_REALLOC: unsafe extern "C" fn(
    *mut u8,
    usize,
    usize,
    usize,
) -> *mut u8 = cabi_realloc;

#[linkage = "weak"]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cabi_realloc(
    old_ptr: *mut u8,
    old_len: usize,
    align: usize,
    new_len: usize,
) -> *mut u8 {
    let layout;
    let ptr = if old_len == 0 {
        if new_len == 0 {
            return ptr::without_provenance_mut(align);
        }
        layout = Layout::from_size_align_unchecked(new_len, align);
        alloc::alloc(layout)
    } else {
        debug_assert_ne!(new_len, 0, "non-zero old_len requires non-zero new_len!");
        layout = Layout::from_size_align_unchecked(old_len, align);
        alloc::realloc(old_ptr, layout, new_len)
    };
    if ptr.is_null() {
        // Print a nice message in debug mode, but in release mode don't
        // pull in so many dependencies related to printing so just emit an
        // `unreachable` instruction.
        if cfg!(debug_assertions) {
            alloc::handle_alloc_error(layout);
        } else {
            super::abort_internal();
        }
    }
    return ptr;
}
