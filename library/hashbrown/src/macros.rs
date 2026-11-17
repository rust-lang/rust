// FIXME: remove once we bump MSRV >= 1.95.0
// Originally by @folkertdev on Zulip:
// https://rust-lang.zulipchat.com/#narrow/channel/219381-t-libs/topic/compiler-builtins.20msrv.20policy.3F/near/606320965
/// Fallback `cfg_select!` on older Rust.
///
/// Requires braces around output, but works for items and expressions.
#[cfg(not(feature = "nightly"))]
#[macro_export]
macro_rules! cfg_select {
    ({ $($tt:tt)* }) => {{
        $crate::cfg_select! { $($tt)* }
    }};
    (_ => { $($output:tt)* }) => {
        $($output)*
    };
    (
        $cfg:meta => $output:tt
        $($( $rest:tt )+)?
    ) => {
        #[cfg($cfg)]
        $crate::cfg_select! { _ => $output }
        $(
            #[cfg(not($cfg))]
            $crate::cfg_select! { $($rest)+ }
        )?
    }
}

// Helper macro for specialization. This also helps avoid parse errors if the
// default fn syntax for specialization changes in the future.
#[cfg(feature = "nightly")]
macro_rules! default_fn {
    (#[$($a:tt)*] $($tt:tt)*) => {
        #[$($a)*] default $($tt)*
    }
}
#[cfg(not(feature = "nightly"))]
macro_rules! default_fn {
    ($($tt:tt)*) => {
        $($tt)*
    }
}
