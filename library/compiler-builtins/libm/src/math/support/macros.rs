/// `libm` cannot have dependencies, so this is vendored directly from the `cfg-if` crate
/// (with some comments stripped for compactness).
macro_rules! cfg_if {
    // match if/else chains with a final `else`
    ($(
        if #[cfg($meta:meta)] { $($tokens:tt)* }
    ) else * else {
        $($tokens2:tt)*
    }) => {
        cfg_if! { @__items () ; $( ( ($meta) ($($tokens)*) ), )* ( () ($($tokens2)*) ), }
    };

    // match if/else chains lacking a final `else`
    (
        if #[cfg($i_met:meta)] { $($i_tokens:tt)* }
        $( else if #[cfg($e_met:meta)] { $($e_tokens:tt)* } )*
    ) => {
        cfg_if! {
            @__items
            () ;
            ( ($i_met) ($($i_tokens)*) ),
            $( ( ($e_met) ($($e_tokens)*) ), )*
            ( () () ),
        }
    };

    // Internal and recursive macro to emit all the items
    //
    // Collects all the negated cfgs in a list at the beginning and after the
    // semicolon is all the remaining items
    (@__items ($($not:meta,)*) ; ) => {};
    (@__items ($($not:meta,)*) ; ( ($($m:meta),*) ($($tokens:tt)*) ), $($rest:tt)*) => {
        #[cfg(all($($m,)* not(any($($not),*))))] cfg_if! { @__identity $($tokens)* }
        cfg_if! { @__items ($($not,)* $($m,)*) ; $($rest)* }
    };

    // Internal macro to make __apply work out right for different match types,
    // because of how macros matching/expand stuff.
    (@__identity $($tokens:tt)*) => { $($tokens)* };
}

/// Choose between using an intrinsic (if available) and the function body. Returns directly if
/// the intrinsic is used, otherwise the rest of the function body is used.
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
