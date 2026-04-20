/// `libm` cannot have dependencies, so this is vendored directly from the `cfg-if` crate
/// (with some comments stripped for compactness).
macro_rules! cfg_if {
    (
        if #[cfg( $($i_meta:tt)+ )] { $( $i_tokens:tt )* }
        $(
            else if #[cfg( $($ei_meta:tt)+ )] { $( $ei_tokens:tt )* }
        )*
        $(
            else { $( $e_tokens:tt )* }
        )?
    ) => {
        cfg_if! {
            @__items () ;
            (( $($i_meta)+ ) ( $( $i_tokens )* )),
            $(
                (( $($ei_meta)+ ) ( $( $ei_tokens )* )),
            )*
            $(
                (() ( $( $e_tokens )* )),
            )?
        }
    };

    // Internal and recursive macro to emit all the items
    //
    // Collects all the previous cfgs in a list at the beginning, so they can be
    // negated. After the semicolon are all the remaining items.
    (@__items ( $( ($($_:tt)*) , )* ) ; ) => {};
    (
        @__items ( $( ($($no:tt)+) , )* ) ;
        (( $( $($yes:tt)+ )? ) ( $( $tokens:tt )* )),
        $( $rest:tt , )*
    ) => {
        // Emit all items within one block, applying an appropriate #[cfg]. The
        // #[cfg] will require all `$yes` matchers specified and must also negate
        // all previous matchers.
        #[cfg(all(
            $( $($yes)+ , )?
            not(any( $( $($no)+ ),* ))
        ))]
        // Subtle: You might think we could put `$( $tokens )*` here. But if
        // that contains multiple items then the `#[cfg(all(..))]` above would
        // only apply to the first one. By wrapping `$( $tokens )*` in this
        // macro call, we temporarily group the items into a single thing (the
        // macro call) that will be included/excluded by the `#[cfg(all(..))]`
        // as appropriate. If the `#[cfg(all(..))]` succeeds, the macro call
        // will be included, and then evaluated, producing `$( $tokens )*`. See
        // also the "issue #90" test below.
        cfg_if! { @__temp_group $( $tokens )* }

        // Recurse to emit all other items in `$rest`, and when we do so add all
        // our `$yes` matchers to the list of `$no` matchers as future emissions
        // will have to negate everything we just matched as well.
        cfg_if! {
            @__items ( $( ($($no)+) , )* $( ($($yes)+) , )? ) ;
            $( $rest , )*
        }
    };

    // See the "Subtle" comment above.
    (@__temp_group $( $tokens:tt )* ) => {
        $( $tokens )*
    };
}

/// Choose between using an arch-specific implementation and the function body. Returns directly
/// if the arch implementation is used, otherwise continue with the rest of the function.
///
/// Specify a `use_arch` meta field if an architecture-specific implementation is provided.
/// These live in the `math::arch::some_target_arch` module.
///
/// Specify a `use_arch_required` meta field if something architecture-specific must be used
/// regardless of feature configuration (`arch`).
///
/// The passed meta options do not need to account for the `arch` target feature.
macro_rules! select_implementation {
    (
        name: $fn_name:ident,
        // Configuration meta for when to use arch-specific implementation that requires hard
        // float ops
        $( use_arch: $use_arch:meta, )?
        // Configuration meta for when to use the arch module regardless of whether softfloats
        // have been requested.
        $( use_arch_required: $use_arch_required:meta, )?
        args: $($arg:ident),+ ,
    ) => {
        // FIXME: these use paths that are a pretty fragile (`super`). We should figure out
        // something better w.r.t. how this is vendored into compiler-builtins.

        // However, we do need a few things from `arch` that are used even with soft floats.
        select_implementation! {
            @cfg $($use_arch_required)?;
            if true {
                return  super::arch::$fn_name( $($arg),+ );
            }
        }

        // By default, never use arch-specific implementations if `arch` is disabled.
        #[cfg(feature = "arch")]
        select_implementation! {
            @cfg $($use_arch)?;
            // Wrap in `if true` to avoid unused warnings
            if true {
                return  super::arch::$fn_name( $($arg),+ );
            }
        }
    };

    // Coalesce helper to construct an expression only if a config is provided
    (@cfg ; $ex:expr) => { };
    (@cfg $provided:meta; $ex:expr) => { #[cfg($provided)] $ex };
}

/// Construct a 16-bit float from hex float representation (C-style), guaranteed to
/// evaluate at compile time.
#[cfg(f16_enabled)]
#[cfg_attr(feature = "unstable-public-internals", macro_export)]
#[allow(unused_macros)]
macro_rules! hf16 {
    ($s:literal) => {{
        const X: f16 = $crate::support::hf16($s);
        X
    }};
}

/// Construct a 32-bit float from hex float representation (C-style), guaranteed to
/// evaluate at compile time.
#[allow(unused_macros)]
#[cfg_attr(feature = "unstable-public-internals", macro_export)]
macro_rules! hf32 {
    ($s:literal) => {{
        const X: f32 = $crate::support::hf32($s);
        X
    }};
}

/// Construct a 64-bit float from hex float representation (C-style), guaranteed to
/// evaluate at compile time.
#[allow(unused_macros)]
#[cfg_attr(feature = "unstable-public-internals", macro_export)]
macro_rules! hf64 {
    ($s:literal) => {{
        const X: f64 = $crate::support::hf64($s);
        X
    }};
}

/// Construct a 128-bit float from hex float representation (C-style), guaranteed to
/// evaluate at compile time.
#[cfg(f128_enabled)]
#[allow(unused_macros)]
#[cfg_attr(feature = "unstable-public-internals", macro_export)]
macro_rules! hf128 {
    ($s:literal) => {{
        const X: f128 = $crate::support::hf128($s);
        X
    }};
}

/// Assert `F::biteq` with better messages.
#[cfg(test)]
macro_rules! assert_biteq {
    ($left:expr, $right:expr, $($tt:tt)*) => {{
        let l = $left;
        let r = $right;
        // hack to get width from a value
        let bits = $crate::support::Int::leading_zeros(l.to_bits() - l.to_bits());
        assert!(
            $crate::support::Float::biteq(l, r),
            "{}\nl: {l:?} ({lb:#0width$x} {lh})\nr: {r:?} ({rb:#0width$x} {rh})",
            format_args!($($tt)*),
            lb = l.to_bits(),
            lh = $crate::support::Hex(l),
            rb = r.to_bits(),
            rh = $crate::support::Hex(r),
            width = ((bits / 4) + 2) as usize,

        );
    }};
    ($left:expr, $right:expr $(,)?) => {
        assert_biteq!($left, $right, "")
    };
}
