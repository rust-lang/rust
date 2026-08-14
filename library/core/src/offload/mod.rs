// offload module
#[unstable(feature = "gpu_offload", issue = "131513")]
pub use crate::macros::builtin::offload_kernel;
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
            args = NONE
        )
    };

    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = NONE; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = (SOME $val); workgroup_dim = $w; thread_dim = $t; dyn_cache = $d; args = $a)
    };
    (@munch [kernel = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = (SOME $old:expr); workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!("duplicate field `kernel`")
    };
    (@munch [workgroup_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = ([1, 1, 1]); thread_dim = $t:tt; dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = (SOME $val); thread_dim = $t; dyn_cache = $d; args = $a)
    };
    (@munch [workgroup_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = (SOME $old:expr); thread_dim = $t:tt; dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!("duplicate field `workgroup_dim`")
    };
    (@munch [thread_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = ([1, 1, 1]); dyn_cache = $d:tt; args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = (SOME $val); dyn_cache = $d; args = $a)
    };
    (@munch [thread_dim = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = (SOME $old:expr); dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!("duplicate field `thread_dim`")
    };
    (@munch [dyn_cache = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = (0); args = $a:tt) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = $t; dyn_cache = (SOME $val); args = $a)
    };
    (@munch [dyn_cache = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = (SOME $old:expr); args = $a:tt) => {
        compile_error!("duplicate field `dyn_cache`")
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = NONE) => {
        $crate::offload!(@munch [$($rest_f = $rest_v),*]; kernel = $k; workgroup_dim = $w; thread_dim = $t; dyn_cache = $d; args = (SOME $val))
    };
    (@munch [args = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = (SOME $old:expr)) => {
        compile_error!("duplicate field `args`")
    };

    (@munch [$invalid:ident = $val:expr $(, $rest_f:ident = $rest_v:expr)*]; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!(concat!("unknown field `", stringify!($invalid), "`"))
    };

    (@munch []; kernel = NONE; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = $a:tt) => {
        compile_error!("missing `kernel`")
    };
    (@munch []; kernel = $k:tt; workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = NONE) => {
        compile_error!("missing `args`")
    };
    (@munch []; kernel = (SOME $kernel:expr); workgroup_dim = $w:tt; thread_dim = $t:tt; dyn_cache = $d:tt; args = (SOME $args:expr)) => {
        $crate::intrinsics::offload::<_, _, ()>(
            $kernel,
            $crate::offload!(@value $w),
            $crate::offload!(@value $t),
            $crate::offload!(@value $d),
            $args,
        )
    };

    (@value (SOME $val:expr)) => { $val };
    (@value ($val:expr)) => { $val };
}
