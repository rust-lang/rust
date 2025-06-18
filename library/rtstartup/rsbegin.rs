// rsbegin.o and rsend.o are the so called "compiler runtime startup objects".
// They contain code needed to correctly initialize the compiler runtime.
//
// When an executable or dylib image is linked, all user code and libraries are
// "sandwiched" between these two object files, so code or data from rsbegin.o
// become first in the respective sections of the image, whereas code and data
// from rsend.o become the last ones. This effect can be used to place symbols
// at the beginning or at the end of a section, as well as to insert any required
// headers or footers.
//
// Note that the actual module entry point is located in the C runtime startup
// object (usually called `crtX.o`), which then invokes initialization callbacks
// of other runtime components (registered via yet another special image section).

#![feature(no_core)]
#![feature(lang_items)]
#![feature(auto_traits)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]
#![allow(internal_features)]
#![warn(unreachable_pub)]

#[cfg(not(bootstrap))]
#[lang = "pointee_sized"]
pub trait PointeeSized {}

#[cfg(not(bootstrap))]
#[lang = "meta_sized"]
pub trait MetaSized: PointeeSized {}

#[cfg(bootstrap)]
#[lang = "sized"]
pub trait Sized {}
#[cfg(not(bootstrap))]
#[lang = "sized"]
pub trait Sized: MetaSized {}

#[lang = "sync"]
auto trait Sync {}
#[lang = "copy"]
trait Copy {}
#[lang = "freeze"]
auto trait Freeze {}

#[cfg(bootstrap)]
impl<T: ?Sized> Copy for *mut T {}
#[cfg(not(bootstrap))]
impl<T: PointeeSized> Copy for *mut T {}

#[cfg(bootstrap)]
#[lang = "drop_in_place"]
#[inline]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    drop_in_place(to_drop);
}
#[cfg(not(bootstrap))]
#[lang = "drop_in_place"]
#[inline]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: PointeeSized>(to_drop: *mut T) {
    drop_in_place(to_drop);
}

// Frame unwind info registration
//
// Each module's image contains a frame unwind info section (usually
// ".eh_frame").  When a module is loaded/unloaded into the process, the
// unwinder must be informed about the location of this section in memory. The
// methods of achieving that vary by the platform.  On some (e.g., Linux), the
// unwinder can discover unwind info sections on its own (by dynamically
// enumerating currently loaded modules via the dl_iterate_phdr() API and
// finding their ".eh_frame" sections); Others, like Windows, require modules
// to actively register their unwind info sections via unwinder API.
#[cfg(all(target_os = "windows", target_arch = "x86", target_env = "gnu"))]
pub mod eh_frames {
    #[no_mangle]
    #[unsafe(link_section = ".eh_frame")]
    // Marks beginning of the stack frame unwind info section
    pub static __EH_FRAME_BEGIN__: [u8; 0] = [];

    // Scratch space for unwinder's internal book-keeping.
    // This is defined as `struct object` in $GCC/libgcc/unwind-dw2-fde.h.
    static mut OBJ: [isize; 6] = [0; 6];

    macro_rules! impl_copy {
        ($($t:ty)*) => {
            $(
                impl ::Copy for $t {}
            )*
        }
    }

    impl_copy! {
        usize u8 u16 u32 u64 u128
        isize i8 i16 i32 i64 i128
        f32 f64
        bool char
    }

    // Unwind info registration/deregistration routines.
    unsafe extern "C" {
        fn __register_frame_info(eh_frame_begin: *const u8, object: *mut u8);
        fn __deregister_frame_info(eh_frame_begin: *const u8, object: *mut u8);
    }

    unsafe extern "C" fn init() {
        // register unwind info on module startup
        __register_frame_info(&__EH_FRAME_BEGIN__ as *const u8, &mut OBJ as *mut _ as *mut u8);
    }

    unsafe extern "C" fn uninit() {
        // unregister on shutdown
        __deregister_frame_info(&__EH_FRAME_BEGIN__ as *const u8, &mut OBJ as *mut _ as *mut u8);
    }

    // MinGW-specific init/uninit routine registration
    pub mod mingw_init {
        // MinGW's startup objects (crt0.o / dllcrt0.o) will invoke global constructors in the
        // .ctors and .dtors sections on startup and exit. In the case of DLLs, this is done when
        // the DLL is loaded and unloaded.
        //
        // The linker will sort the sections, which ensures that our callbacks are located at the
        // end of the list. Since constructors are run in reverse order, this ensures that our
        // callbacks are the first and last ones executed.

        #[unsafe(link_section = ".ctors.65535")] // .ctors.* : C initialization callbacks
        pub static P_INIT: unsafe extern "C" fn() = super::init;

        #[unsafe(link_section = ".dtors.65535")] // .dtors.* : C termination callbacks
        pub static P_UNINIT: unsafe extern "C" fn() = super::uninit;
    }
}
