/// Choose among using an intrinsic (if available) and falling back to the default function body.
/// Returns directly if the intrinsic version is used, otherwise continues to the rest of the
/// function.
///
/// Use this if the intrinsic is likely to be more performant on the platform(s) specified
/// in `intrinsic_available`.
///
/// The `cfg` used here is controlled by `build.rs` so the passed meta does not need to account
/// for e.g. the `unstable-intrinsics` or `force-soft-float` features.
macro_rules! select_implementation {
    (
        name: $fname:ident,
        // Configuration meta for when to call intrinsics and let LLVM figure it out
        $( use_intrinsic: $use_intrinsic:meta, )?
        args: $($arg:ident),+ ,
    ) => {
        // FIXME: these use paths that are a pretty fragile (`super`). We should figure out
        // something better w.r.t. how this is vendored into compiler-builtins.

        // Never use intrinsics if we are forcing soft floats, and only enable with the
        // `unstable-intrinsics` feature.
        #[cfg(intrinsics_enabled)]
        select_implementation! {
            @cfg $( $use_intrinsic )?;
            if true {
                return  super::arch::intrinsics::$fname( $($arg),+ );
            }
        }
    };

    // Coalesce helper to construct an expression only if a config is provided
    (@cfg ; $ex:expr) => { };
    (@cfg $provided:meta; $ex:expr) => { #[cfg($provided)] $ex };
}
