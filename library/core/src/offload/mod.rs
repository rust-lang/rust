// offload module
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::macros::builtin::offload_kernel;
use crate::marker::PhantomData;
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::offload;

/// Launches a kernel on an offload device (e.g., a GPU).
///
/// This macro is an interface over the `offload` intrinsic. The kernel itself must be defined
/// using the [`offload_kernel`] macro.
///
/// The following named arguments are accepted:
///
/// - `kernel`: The kernel function to offload. Must be a function item. (required)
/// - `args`: A tuple of arguments forwarded to `kernel`. (required)
/// - `workgroup_dim`: A 3D size specifying the number of workgroups to launch.
///   Defaults to `[1, 1, 1]`.
/// - `thread_dim`: A 3D size specifying the number of threads per workgroup.
///   Defaults to `[1, 1, 1]`.
/// - `dyn_cache`: The amount of dynamic shared memory, in bytes, to allocate for the kernel.
///   Defaults to `0`.
/// - `device`: The index of the device to offload to. Must be `>= 0`. If omitted, the
///   default device is used. Use [`crate::intrinsics::offload_get_num_devices`] to discover
///   which device ids are valid.
///
/// Each argument may only be specified once.
///
/// # Examples
///
/// ```rust,ignore (offload requires a -Z flag)
/// let mut x = [0.0f64; 256];
/// core::offload::offload! {
///     kernel = kernel,
///     workgroup_dim = [256, 1, 1],
///     args = (&mut x as *mut [f64; 256],),
/// }
/// ```
#[macro_export]
#[unstable(feature = "gpu_offload", issue = "131513")]
#[allow_internal_unstable(core_intrinsics)]
macro_rules! offload {
    ( $($field:ident = $val:expr),* $(,)? ) => {
        $crate::offload!(@munch
            [ $($field = $val),* ];
            kernel = NONE;
            workgroup_dim = ([1, 1, 1]);
            thread_dim = ([1, 1, 1]);
            dyn_cache = (0);
            device = NONE;
            args = NONE
        )
    };

    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = NONE; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = (SOME $val); workgroup_dim = $w; thread_dim = $t; dyn_cache = $d; device = $device; args = $a)
    };
    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = (SOME $old:expr); workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        compile_error!("duplicate field `kernel`")
    };
    (@munch [workgroup_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = ([1, 1, 1]); thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = (SOME $val); thread_dim = $t; dyn_cache = $d; device = $device; args = $a)
    };
    (@munch [workgroup_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = (SOME $old:expr); thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        compile_error!("duplicate field `workgroup_dim`")
    };
    (@munch [thread_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = ([1, 1, 1]); dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = (SOME $val); dyn_cache = $d; device = $device; args = $a)
    };
    (@munch [thread_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = (SOME $old:expr); dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        compile_error!("duplicate field `thread_dim`")
    };
    (@munch [dyn_cache = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = (0); device = $device:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = $t; dyn_cache = (SOME $val); device = $device; args = $a)
    };
    (@munch [dyn_cache = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = (SOME $old:expr); device = $device:tt; args = $a:tt) => {
        compile_error!("duplicate field `dyn_cache`")
    };
    (@munch [device = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = NONE; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = $t; dyn_cache = $d; device = (SOME $val); args = $a)
    };
    (@munch [device = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = (SOME $old:expr); args = $a:tt) => {
        compile_error!("duplicate field `device`")
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = NONE) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = $t; dyn_cache = $d; device = $device; args = (SOME $val))
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = (SOME $old:expr)) => {
        compile_error!("duplicate field `args`")
    };

    (@munch [$invalid:ident = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        compile_error!(concat!("unknown field `", stringify!($invalid), "`"))
    };

    (@munch []; kernel = NONE; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = $a:tt) => {
        compile_error!("missing `kernel`")
    };
    (@munch []; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = NONE) => {
        compile_error!("missing `args`")
    };
    (@munch []; kernel = (SOME $kernel:expr); workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; device = $device:tt; args = (SOME $args:expr)) => {
        $crate::intrinsics::offload::<_, _, ()>(
            $kernel,
            $crate::offload!(@value $w),
            $crate::offload!(@value $t),
            $crate::offload!(@value $d),
            $crate::offload!(@device $device),
            $args,
        )
    };

    (@value (SOME $val:expr)) => { $val };
    (@value ($val:expr)) => { $val };

    // if `device` is omitted (`NONE), we use the OpenMP default device (`-1`)
    (@device NONE) => { -1 };
    (@device (SOME $val:expr)) => { {
        const { $crate::assert!($val >= 0, "offload device must be non-negative; omit `device` to use the default device") };
        let device: i32 = $val;
        $crate::assert!(
            device < $crate::intrinsics::offload_get_num_devices(),
            "offload device {} is not available",
            device,
        );
        device
    } };
}

// Region & Partitioning Strategy

/// Defines how execution units access memory regions.
///
/// # Safety
///
/// Implementations must guarantee that generated views are disjoint.
#[unstable(feature = "offload", issue = "124509")]
pub unsafe trait PartitioningStrategy {
    /// Read-only view type for the partitioned memory region.
    type View<'a, T: 'a>;

    /// Mutable view type for the partitioned memory region.
    type ViewMut<'a, T: 'a>;

    /// Returns the execution index of the current unit.
    fn index() -> usize;

    /// Returns a read-only view of the region for the current execution context.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `len` valid, initialized elements of type `T`.
    /// The memory must stay valid for lifetime `'a`.
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>>;

    /// Returns a mutable view of the region for the current execution context.
    ///
    /// # Safety
    ///
    /// `ptr` must point to `len` valid, initialized elements of type `T`.
    /// The memory must stay valid for lifetime `'a`.
    /// The returned view must be disjoint from all other active views.
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>>;
}

/// A memory region bound to a partitioning strategy.
#[derive(Copy, Clone)]
#[unstable(feature = "offload", issue = "124509")]
pub struct Region<'a, T, S: PartitioningStrategy> {
    ptr: *mut T,
    len: usize,
    _marker: core::marker::PhantomData<(&'a mut [T], S)>,
}

/// Raw representation used to build a [`Region`] from common aggregate types.
struct RawRegion<'a, T> {
    pub ptr: *mut T,
    pub len: usize,
    _marker: core::marker::PhantomData<&'a mut [T]>,
}

impl<'a, T> From<&'a mut [T]> for RawRegion<'a, T> {
    fn from(data: &'a mut [T]) -> Self {
        Self { ptr: data.as_mut_ptr(), len: data.len(), _marker: core::marker::PhantomData }
    }
}

impl<'a, T, const N: usize> From<&'a mut [T; N]> for RawRegion<'a, T> {
    fn from(data: &'a mut [T; N]) -> Self {
        Self { ptr: data.as_mut_ptr(), len: N, _marker: core::marker::PhantomData }
    }
}

#[unstable(feature = "offload", issue = "124509")]
impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    /// Creates a new partitioned region from data convertible into a [`RawRegion`].
    pub fn new<D>(data: D) -> Self
    where
        D: Into<RawRegion<'a, T>>,
    {
        let raw = data.into();
        Self { ptr: raw.ptr, len: raw.len, _marker: core::marker::PhantomData }
    }

    /// Returns a read-only view for the current execution context.
    pub fn get(&self) -> Option<S::View<'_, T>> {
        // SAFETY: `self.ptr` points to `self.len` valid elements for lifetime `'a`.
        unsafe { S::get(self.ptr as *const T, self.len) }
    }

    /// Returns a mutable view for the current execution context.
    pub fn get_mut(&mut self) -> Option<S::ViewMut<'_, T>> {
        // SAFETY: `self.ptr` points to `self.len` valid elements for lifetime `'a`.
        // The strategy guarantees that the returned view is disjoint.
        unsafe { S::get_mut(self.ptr, self.len) }
    }
}
