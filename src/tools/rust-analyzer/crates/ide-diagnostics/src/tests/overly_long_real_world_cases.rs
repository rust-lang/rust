//! Overly long excerpts of failures from real world cases, that I was too lazy to minimize.

use crate::tests::check_diagnostics_with_disabled;

#[test]
fn tracing_infinite_repeat() {
    check_diagnostics_with_disabled(
        r#"
//- /core.rs crate:core
#[rustc_builtin_macro]
#[macro_export]
macro_rules! concat {
($($e:expr),* $(,)?) => {{ /* compiler built-in */ }};
}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! file {
() => {
    /* compiler built-in */
};
}
#[allow_internal_unsafe]
#[allow_internal_unstable(fmt_internals)]
#[rustc_builtin_macro]
#[macro_export]
macro_rules! format_args {
($fmt:expr) => {{ /* compiler built-in */ }};
($fmt:expr, $($args:tt)*) => {{ /* compiler built-in */ }};
}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! line {
() => {
    /* compiler built-in */
};
}

//- /tracing_core.rs crate:tracing_core deps:core
#[macro_export]
macro_rules! identify_callsite {
($callsite:expr) => {
    $crate::callsite::Identifier($callsite)
};
}

#[macro_export]
macro_rules! metadata {
(
    name: $name:expr,
    target: $target:expr,
    level: $level:expr,
    fields: $fields:expr,
    callsite: $callsite:expr,
    kind: $kind:expr
) => {
    $crate::metadata! {
        name: $name,
        target: $target,
        level: $level,
        fields: $fields,
        callsite: $callsite,
        kind: $kind,
    }
};
(
    name: $name:expr,
    target: $target:expr,
    level: $level:expr,
    fields: $fields:expr,
    callsite: $callsite:expr,
    kind: $kind:expr,
) => {
    $crate::metadata::Metadata::new(
        $name,
        $target,
        $level,
        $crate::__macro_support::Option::Some($crate::__macro_support::file!()),
        $crate::__macro_support::Option::Some($crate::__macro_support::line!()),
        $crate::__macro_support::Option::Some($crate::__macro_support::module_path!()),
        $crate::field::FieldSet::new($fields, $crate::identify_callsite!($callsite)),
        $kind,
    )
};
}

//- /tracing.rs crate:tracing deps:core,tracing_core
#[doc(hidden)]
pub mod __macro_support {
// Re-export the `core` functions that are used in macros. This allows
// a crate to be named `core` and avoid name clashes.
// See here: https://github.com/tokio-rs/tracing/issues/2761
pub use core::{concat, file, format_args, iter::Iterator, line, option::Option};
}

#[macro_export]
macro_rules! span {
(target: $target:expr, parent: $parent:expr, $lvl:expr, $name:expr) => {
    $crate::span!(target: $target, parent: $parent, $lvl, $name,)
};
(target: $target:expr, parent: $parent:expr, $lvl:expr, $name:expr, $($fields:tt)*) => {
    {
        use $crate::__macro_support::Callsite as _;
        static __CALLSITE: $crate::__macro_support::MacroCallsite = $crate::callsite2! {
            name: $name,
            kind: $crate::metadata::Kind::SPAN,
            target: $target,
            level: $lvl,
            fields: $($fields)*
        };
        let mut interest = $crate::subscriber::Interest::never();
        if $crate::level_enabled!($lvl)
            && { interest = __CALLSITE.interest(); !interest.is_never() }
            && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
        {
            let meta = __CALLSITE.metadata();
            // span with explicit parent
            $crate::Span::child_of(
                $parent,
                meta,
                &$crate::valueset!(meta.fields(), $($fields)*),
            )
        } else {
            let span = $crate::__macro_support::__disabled_span(__CALLSITE.metadata());
            $crate::if_log_enabled! { $lvl, {
                span.record_all(&$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
            }};
            span
        }
    }
};
(target: $target:expr, $lvl:expr, $name:expr, $($fields:tt)*) => {
    {
        use $crate::__macro_support::Callsite as _;
        static __CALLSITE: $crate::callsite::DefaultCallsite = $crate::callsite2! {
            name: $name,
            kind: $crate::metadata::Kind::SPAN,
            target: $target,
            level: $lvl,
            fields: $($fields)*
        };
        let mut interest = $crate::subscriber::Interest::never();
        if $crate::level_enabled!($lvl)
            && { interest = __CALLSITE.interest(); !interest.is_never() }
            && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
        {
            let meta = __CALLSITE.metadata();
            // span with contextual parent
            $crate::Span::new(
                meta,
                &$crate::valueset!(meta.fields(), $($fields)*),
            )
        } else {
            let span = $crate::__macro_support::__disabled_span(__CALLSITE.metadata());
            $crate::if_log_enabled! { $lvl, {
                span.record_all(&$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
            }};
            span
        }
    }
};
(target: $target:expr, parent: $parent:expr, $lvl:expr, $name:expr) => {
    $crate::span!(target: $target, parent: $parent, $lvl, $name,)
};
(parent: $parent:expr, $lvl:expr, $name:expr, $($fields:tt)*) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        $name,
        $($fields)*
    )
};
(parent: $parent:expr, $lvl:expr, $name:expr) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        $name,
    )
};
(target: $target:expr, $lvl:expr, $name:expr, $($fields:tt)*) => {
    $crate::span!(
        target: $target,
        $lvl,
        $name,
        $($fields)*
    )
};
(target: $target:expr, $lvl:expr, $name:expr) => {
    $crate::span!(target: $target, $lvl, $name,)
};
($lvl:expr, $name:expr, $($fields:tt)*) => {
    $crate::span!(
        target: module_path!(),
        $lvl,
        $name,
        $($fields)*
    )
};
($lvl:expr, $name:expr) => {
    $crate::span!(
        target: module_path!(),
        $lvl,
        $name,
    )
};
}

#[macro_export]
macro_rules! trace_span {
(target: $target:expr, parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        parent: $parent,
        $crate::Level::TRACE,
        $name,
        $($field)*
    )
};
(target: $target:expr, parent: $parent:expr, $name:expr) => {
    $crate::trace_span!(target: $target, parent: $parent, $name,)
};
(parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        $name,
        $($field)*
    )
};
(parent: $parent:expr, $name:expr) => {
    $crate::trace_span!(parent: $parent, $name,)
};
(target: $target:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        $crate::Level::TRACE,
        $name,
        $($field)*
    )
};
(target: $target:expr, $name:expr) => {
    $crate::trace_span!(target: $target, $name,)
};
($name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        $crate::Level::TRACE,
        $name,
        $($field)*
    )
};
($name:expr) => { $crate::trace_span!($name,) };
}

#[macro_export]
macro_rules! debug_span {
(target: $target:expr, parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        parent: $parent,
        $crate::Level::DEBUG,
        $name,
        $($field)*
    )
};
(target: $target:expr, parent: $parent:expr, $name:expr) => {
    $crate::debug_span!(target: $target, parent: $parent, $name,)
};
(parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        $name,
        $($field)*
    )
};
(parent: $parent:expr, $name:expr) => {
    $crate::debug_span!(parent: $parent, $name,)
};
(target: $target:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        $crate::Level::DEBUG,
        $name,
        $($field)*
    )
};
(target: $target:expr, $name:expr) => {
    $crate::debug_span!(target: $target, $name,)
};
($name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        $crate::Level::DEBUG,
        $name,
        $($field)*
    )
};
($name:expr) => {$crate::debug_span!($name,)};
}

#[macro_export]
macro_rules! info_span {
(target: $target:expr, parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        parent: $parent,
        $crate::Level::INFO,
        $name,
        $($field)*
    )
};
(target: $target:expr, parent: $parent:expr, $name:expr) => {
    $crate::info_span!(target: $target, parent: $parent, $name,)
};
(parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        $name,
        $($field)*
    )
};
(parent: $parent:expr, $name:expr) => {
    $crate::info_span!(parent: $parent, $name,)
};
(target: $target:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        $crate::Level::INFO,
        $name,
        $($field)*
    )
};
(target: $target:expr, $name:expr) => {
    $crate::info_span!(target: $target, $name,)
};
($name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        $crate::Level::INFO,
        $name,
        $($field)*
    )
};
($name:expr) => {$crate::info_span!($name,)};
}

#[macro_export]
macro_rules! warn_span {
(target: $target:expr, parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        parent: $parent,
        $crate::Level::WARN,
        $name,
        $($field)*
    )
};
(target: $target:expr, parent: $parent:expr, $name:expr) => {
    $crate::warn_span!(target: $target, parent: $parent, $name,)
};
(parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        $name,
        $($field)*
    )
};
(parent: $parent:expr, $name:expr) => {
    $crate::warn_span!(parent: $parent, $name,)
};
(target: $target:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        $crate::Level::WARN,
        $name,
        $($field)*
    )
};
(target: $target:expr, $name:expr) => {
    $crate::warn_span!(target: $target, $name,)
};
($name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        $crate::Level::WARN,
        $name,
        $($field)*
    )
};
($name:expr) => {$crate::warn_span!($name,)};
}

#[macro_export]
macro_rules! error_span {
(target: $target:expr, parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        parent: $parent,
        $crate::Level::ERROR,
        $name,
        $($field)*
    )
};
(target: $target:expr, parent: $parent:expr, $name:expr) => {
    $crate::error_span!(target: $target, parent: $parent, $name,)
};
(parent: $parent:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        $name,
        $($field)*
    )
};
(parent: $parent:expr, $name:expr) => {
    $crate::error_span!(parent: $parent, $name,)
};
(target: $target:expr, $name:expr, $($field:tt)*) => {
    $crate::span!(
        target: $target,
        $crate::Level::ERROR,
        $name,
        $($field)*
    )
};
(target: $target:expr, $name:expr) => {
    $crate::error_span!(target: $target, $name,)
};
($name:expr, $($field:tt)*) => {
    $crate::span!(
        target: module_path!(),
        $crate::Level::ERROR,
        $name,
        $($field)*
    )
};
($name:expr) => {$crate::error_span!($name,)};
}

#[macro_export]
macro_rules! event {
// Name / target / parent.
(name: $name:expr, target: $target:expr, parent: $parent:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    use $crate::__macro_support::Callsite as _;
    static __CALLSITE: $crate::__macro_support::MacroCallsite = $crate::callsite2! {
        name: $name,
        kind: $crate::metadata::Kind::EVENT,
        target: $target,
        level: $lvl,
        fields: $($fields)*
    };

    let enabled = $crate::level_enabled!($lvl) && {
        let interest = __CALLSITE.interest();
        !interest.is_never() && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
    };
    if enabled {
        (|value_set: $crate::field::ValueSet| {
            $crate::__tracing_log!(
                $lvl,
                __CALLSITE,
                &value_set
            );
            let meta = __CALLSITE.metadata();
            // event with explicit parent
            $crate::Event::child_of(
                $parent,
                meta,
                &value_set
            );
        })($crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
    } else {
        $crate::__tracing_log!(
            $lvl,
            __CALLSITE,
            &$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*)
        );
    }
});
(name: $name:expr, target: $target:expr, parent: $parent:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        name: $name,
        target: $target,
        parent: $parent,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $lvl:expr, $($k:ident).+ = $($fields:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $lvl, { $($k).+ = $($fields)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $lvl:expr, $($arg:tt)+) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $lvl, { $($arg)+ })
);

// Name / target.
(name: $name:expr, target: $target:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    use $crate::__macro_support::Callsite as _;
    static __CALLSITE: $crate::__macro_support::MacroCallsite = $crate::callsite2! {
        name: $name,
        kind: $crate::metadata::Kind::EVENT,
        target: $target,
        level: $lvl,
        fields: $($fields)*
    };
    let enabled = $crate::level_enabled!($lvl) && {
        let interest = __CALLSITE.interest();
        !interest.is_never() && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
    };
    if enabled {
        (|value_set: $crate::field::ValueSet| {
            let meta = __CALLSITE.metadata();
            // event with contextual parent
            $crate::Event::dispatch(
                meta,
                &value_set
            );
            $crate::__tracing_log!(
                $lvl,
                __CALLSITE,
                &value_set
            );
        })($crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
    } else {
        $crate::__tracing_log!(
            $lvl,
            __CALLSITE,
            &$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*)
        );
    }
});
(name: $name:expr, target: $target:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        name: $name,
        target: $target,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(name: $name:expr, target: $target:expr, $lvl:expr, $($k:ident).+ = $($fields:tt)* ) => (
    $crate::event!(name: $name, target: $target, $lvl, { $($k).+ = $($fields)* })
);
(name: $name:expr, target: $target:expr, $lvl:expr, $($arg:tt)+) => (
    $crate::event!(name: $name, target: $target, $lvl, { $($arg)+ })
);

// Target / parent.
(target: $target:expr, parent: $parent:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    use $crate::__macro_support::Callsite as _;
    static __CALLSITE: $crate::callsite::DefaultCallsite = $crate::callsite2! {
        name: $crate::__macro_support::concat!(
            "event ",
            $crate::__macro_support::file!(),
            ":",
            $crate::__macro_support::line!()
        ),
        kind: $crate::metadata::Kind::EVENT,
        target: $target,
        level: $lvl,
        fields: $($fields)*
    };

    let enabled = $crate::level_enabled!($lvl) && {
        let interest = __CALLSITE.interest();
        !interest.is_never() && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
    };
    if enabled {
        (|value_set: $crate::field::ValueSet| {
            $crate::__tracing_log!(
                $lvl,
                __CALLSITE,
                &value_set
            );
            let meta = __CALLSITE.metadata();
            // event with explicit parent
            $crate::Event::child_of(
                $parent,
                meta,
                &value_set
            );
        })($crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
    } else {
        $crate::__tracing_log!(
            $lvl,
            __CALLSITE,
            &$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*)
        );
    }
});
(target: $target:expr, parent: $parent:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        target: $target,
        parent: $parent,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(target: $target:expr, parent: $parent:expr, $lvl:expr, $($k:ident).+ = $($fields:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $lvl, { $($k).+ = $($fields)* })
);
(target: $target:expr, parent: $parent:expr, $lvl:expr, $($arg:tt)+) => (
    $crate::event!(target: $target, parent: $parent, $lvl, { $($arg)+ })
);

// Name / parent.
(name: $name:expr, parent: $parent:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    use $crate::__macro_support::Callsite as _;
    static __CALLSITE: $crate::__macro_support::MacroCallsite = $crate::callsite2! {
        name: $name,
        kind: $crate::metadata::Kind::EVENT,
        target: module_path!(),
        level: $lvl,
        fields: $($fields)*
    };

    let enabled = $crate::level_enabled!($lvl) && {
        let interest = __CALLSITE.interest();
        !interest.is_never() && __CALLSITE.is_enabled(interest)
    };
    if enabled {
        (|value_set: $crate::field::ValueSet| {
            $crate::__tracing_log!(
                $lvl,
                __CALLSITE,
                &value_set
            );
            let meta = __CALLSITE.metadata();
            // event with explicit parent
            $crate::Event::child_of(
                $parent,
                meta,
                &value_set
            );
        })($crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
    } else {
        $crate::__tracing_log!(
            $lvl,
            __CALLSITE,
            &$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*)
        );
    }
});
(name: $name:expr, parent: $parent:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        name: $name,
        parent: $parent,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(name: $name:expr, parent: $parent:expr, $lvl:expr, $($k:ident).+ = $($fields:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $lvl, { $($k).+ = $($fields)* })
);
(name: $name:expr, parent: $parent:expr, $lvl:expr, $($arg:tt)+) => (
    $crate::event!(name: $name, parent: $parent, $lvl, { $($arg)+ })
);

// Name.
(name: $name:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    use $crate::__macro_support::Callsite as _;
    static __CALLSITE: $crate::__macro_support::MacroCallsite = $crate::callsite2! {
        name: $name,
        kind: $crate::metadata::Kind::EVENT,
        target: module_path!(),
        level: $lvl,
        fields: $($fields)*
    };
    let enabled = $crate::level_enabled!($lvl) && {
        let interest = __CALLSITE.interest();
        !interest.is_never() && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
    };
    if enabled {
        (|value_set: $crate::field::ValueSet| {
            let meta = __CALLSITE.metadata();
            // event with contextual parent
            $crate::Event::dispatch(
                meta,
                &value_set
            );
            $crate::__tracing_log!(
                $lvl,
                __CALLSITE,
                &value_set
            );
        })($crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
    } else {
        $crate::__tracing_log!(
            $lvl,
            __CALLSITE,
            &$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*)
        );
    }
});
(name: $name:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        name: $name,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(name: $name:expr, $lvl:expr, $($k:ident).+ = $($fields:tt)* ) => (
    $crate::event!(name: $name, $lvl, { $($k).+ = $($fields)* })
);
(name: $name:expr, $lvl:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, $lvl, { $($arg)+ })
);

// Target.
(target: $target:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    use $crate::__macro_support::Callsite as _;
    static __CALLSITE: $crate::callsite::DefaultCallsite = $crate::callsite2! {
        name: $crate::__macro_support::concat!(
            "event ",
            $crate::__macro_support::file!(),
            ":",
            $crate::__macro_support::line!()
        ),
        kind: $crate::metadata::Kind::EVENT,
        target: $target,
        level: $lvl,
        fields: $($fields)*
    };
    let enabled = $crate::level_enabled!($lvl) && {
        let interest = __CALLSITE.interest();
        !interest.is_never() && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest)
    };
    if enabled {
        (|value_set: $crate::field::ValueSet| {
            let meta = __CALLSITE.metadata();
            // event with contextual parent
            $crate::Event::dispatch(
                meta,
                &value_set
            );
            $crate::__tracing_log!(
                $lvl,
                __CALLSITE,
                &value_set
            );
        })($crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*));
    } else {
        $crate::__tracing_log!(
            $lvl,
            __CALLSITE,
            &$crate::valueset!(__CALLSITE.metadata().fields(), $($fields)*)
        );
    }
});
(target: $target:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        target: $target,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(target: $target:expr, $lvl:expr, $($k:ident).+ = $($fields:tt)* ) => (
    $crate::event!(target: $target, $lvl, { $($k).+ = $($fields)* })
);
(target: $target:expr, $lvl:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, $lvl, { $($arg)+ })
);

// Parent.
(parent: $parent:expr, $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
(parent: $parent:expr, $lvl:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { $($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $lvl:expr, ?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { ?$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $lvl:expr, %$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { %$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $lvl:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { $($k).+, $($field)*}
    )
);
(parent: $parent:expr, $lvl:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { %$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $lvl:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $lvl,
        { ?$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $lvl:expr, $($arg:tt)+ ) => (
    $crate::event!(target: module_path!(), parent: $parent, $lvl, { $($arg)+ })
);

// ...
( $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $lvl,
        { message = $crate::__macro_support::format_args!($($arg)+), $($fields)* }
    )
);
( $lvl:expr, { $($fields:tt)* }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $lvl,
        { message = format_args!($($arg)+), $($fields)* }
    )
);
($lvl:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $lvl,
        { $($k).+ = $($field)*}
    )
);
($lvl:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $lvl,
        { $($k).+, $($field)*}
    )
);
($lvl:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $lvl,
        { ?$($k).+, $($field)*}
    )
);
($lvl:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $lvl,
        { %$($k).+, $($field)*}
    )
);
($lvl:expr, ?$($k:ident).+) => (
    $crate::event!($lvl, ?$($k).+,)
);
($lvl:expr, %$($k:ident).+) => (
    $crate::event!($lvl, %$($k).+,)
);
($lvl:expr, $($k:ident).+) => (
    $crate::event!($lvl, $($k).+,)
);
( $lvl:expr, $($arg:tt)+ ) => (
    $crate::event!(target: module_path!(), $lvl, { $($arg)+ })
);
}

#[macro_export]
macro_rules! event_enabled {
($($rest:tt)*)=> (
    $crate::enabled!(kind: $crate::metadata::Kind::EVENT, $($rest)*)
)
}

#[macro_export]
macro_rules! span_enabled {
($($rest:tt)*)=> (
    $crate::enabled!(kind: $crate::metadata::Kind::SPAN, $($rest)*)
)
}

#[macro_export]
macro_rules! enabled {
(kind: $kind:expr, target: $target:expr, $lvl:expr, { $($fields:tt)* } )=> ({
    if $crate::level_enabled!($lvl) {
        use $crate::__macro_support::Callsite as _;
        static __CALLSITE: $crate::callsite::DefaultCallsite = $crate::callsite2! {
            name: $crate::__macro_support::concat!(
                "enabled ",
                $crate::__macro_support::file!(),
                ":",
                $crate::__macro_support::line!()
            ),
            kind: $kind.hint(),
            target: $target,
            level: $lvl,
            fields: $($fields)*
        };
        let interest = __CALLSITE.interest();
        if !interest.is_never() && $crate::__macro_support::__is_enabled(__CALLSITE.metadata(), interest) {
            let meta = __CALLSITE.metadata();
            $crate::dispatcher::get_default(|current| current.enabled(meta))
        } else {
            false
        }
    } else {
        false
    }
});
// Just target and level
(kind: $kind:expr, target: $target:expr, $lvl:expr ) => (
    $crate::enabled!(kind: $kind, target: $target, $lvl, { })
);
(target: $target:expr, $lvl:expr ) => (
    $crate::enabled!(kind: $crate::metadata::Kind::HINT, target: $target, $lvl, { })
);

// These four cases handle fields with no values
(kind: $kind:expr, target: $target:expr, $lvl:expr, $($field:tt)*) => (
    $crate::enabled!(
        kind: $kind,
        target: $target,
        $lvl,
        { $($field)*}
    )
);
(target: $target:expr, $lvl:expr, $($field:tt)*) => (
    $crate::enabled!(
        kind: $crate::metadata::Kind::HINT,
        target: $target,
        $lvl,
        { $($field)*}
    )
);

// Level and field case
(kind: $kind:expr, $lvl:expr, $($field:tt)*) => (
    $crate::enabled!(
        kind: $kind,
        target: module_path!(),
        $lvl,
        { $($field)*}
    )
);

// Simplest `enabled!` case
(kind: $kind:expr, $lvl:expr) => (
    $crate::enabled!(kind: $kind, target: module_path!(), $lvl, { })
);
($lvl:expr) => (
    $crate::enabled!(kind: $crate::metadata::Kind::HINT, target: module_path!(), $lvl, { })
);

// Fallthrough from above
($lvl:expr, $($field:tt)*) => (
    $crate::enabled!(
        kind: $crate::metadata::Kind::HINT,
        target: module_path!(),
        $lvl,
        { $($field)*}
    )
);
}

#[macro_export]
macro_rules! trace {
// Name / target / parent.
(name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::TRACE, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::TRACE, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::TRACE, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::TRACE, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::TRACE, {}, $($arg)+)
);

// Name / target.
(name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::TRACE, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::TRACE, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::TRACE, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::TRACE, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::TRACE, {}, $($arg)+)
);

// Target / parent.
(target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::TRACE, { $($field)* }, $($arg)*)
);
(target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::TRACE, { $($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::TRACE, { ?$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::TRACE, { %$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::TRACE, {}, $($arg)+)
);

// Name / parent.
(name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::TRACE, { $($field)* }, $($arg)*)
);
(name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::TRACE, { $($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::TRACE, { ?$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::TRACE, { %$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::TRACE, {}, $($arg)+)
);

// Name.
(name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::TRACE, { $($field)* }, $($arg)*)
);
(name: $name:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::TRACE, { $($k).+ $($field)* })
);
(name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::TRACE, { ?$($k).+ $($field)* })
);
(name: $name:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::TRACE, { %$($k).+ $($field)* })
);
(name: $name:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, $crate::Level::TRACE, {}, $($arg)+)
);

// Target.
(target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::TRACE, { $($field)* }, $($arg)*)
);
(target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::TRACE, { $($k).+ $($field)* })
);
(target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::TRACE, { ?$($k).+ $($field)* })
);
(target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::TRACE, { %$($k).+ $($field)* })
);
(target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, $crate::Level::TRACE, {}, $($arg)+)
);

// Parent.
(parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { $($field)+ },
        $($arg)+
    )
);
(parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { $($k).+ = $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { ?$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { %$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { $($k).+, $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { ?$($k).+, $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        { %$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::TRACE,
        {},
        $($arg)+
    )
);

// ...
({ $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { $($field)+ },
        $($arg)+
    )
);
($($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { $($k).+ = $($field)*}
    )
);
(?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { ?$($k).+ = $($field)*}
    )
);
(%$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { %$($k).+ = $($field)*}
    )
);
($($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { $($k).+, $($field)*}
    )
);
(?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { ?$($k).+, $($field)*}
    )
);
(%$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { %$($k).+, $($field)*}
    )
);
(?$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { ?$($k).+ }
    )
);
(%$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { %$($k).+ }
    )
);
($($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        { $($k).+ }
    )
);
($($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::TRACE,
        $($arg)+
    )
);
}

#[macro_export]
macro_rules! debug {
// Name / target / parent.
(name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::DEBUG, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::DEBUG, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::DEBUG, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::DEBUG, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::DEBUG, {}, $($arg)+)
);

// Name / target.
(name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::DEBUG, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::DEBUG, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::DEBUG, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::DEBUG, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::DEBUG, {}, $($arg)+)
);

// Target / parent.
(target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::DEBUG, { $($field)* }, $($arg)*)
);
(target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::DEBUG, { $($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::DEBUG, { ?$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::DEBUG, { %$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::DEBUG, {}, $($arg)+)
);

// Name / parent.
(name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::DEBUG, { $($field)* }, $($arg)*)
);
(name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::DEBUG, { $($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::DEBUG, { ?$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::DEBUG, { %$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::DEBUG, {}, $($arg)+)
);

// Name.
(name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::DEBUG, { $($field)* }, $($arg)*)
);
(name: $name:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::DEBUG, { $($k).+ $($field)* })
);
(name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::DEBUG, { ?$($k).+ $($field)* })
);
(name: $name:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::DEBUG, { %$($k).+ $($field)* })
);
(name: $name:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, $crate::Level::DEBUG, {}, $($arg)+)
);

// Target.
(target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::DEBUG, { $($field)* }, $($arg)*)
);
(target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::DEBUG, { $($k).+ $($field)* })
);
(target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::DEBUG, { ?$($k).+ $($field)* })
);
(target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::DEBUG, { %$($k).+ $($field)* })
);
(target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, $crate::Level::DEBUG, {}, $($arg)+)
);

// Parent.
(parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { $($field)+ },
        $($arg)+
    )
);
(parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { $($k).+ = $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { ?$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { %$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { $($k).+, $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { ?$($k).+, $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        { %$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::DEBUG,
        {},
        $($arg)+
    )
);

// ...
({ $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { $($field)+ },
        $($arg)+
    )
);
($($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { $($k).+ = $($field)*}
    )
);
(?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { ?$($k).+ = $($field)*}
    )
);
(%$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { %$($k).+ = $($field)*}
    )
);
($($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { $($k).+, $($field)*}
    )
);
(?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { ?$($k).+, $($field)*}
    )
);
(%$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { %$($k).+, $($field)*}
    )
);
(?$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { ?$($k).+ }
    )
);
(%$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { %$($k).+ }
    )
);
($($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        { $($k).+ }
    )
);
($($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::DEBUG,
        $($arg)+
    )
);
}

#[macro_export]
macro_rules! info {
// Name / target / parent.
(name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::INFO, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::INFO, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::INFO, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::INFO, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::INFO, {}, $($arg)+)
);

// Name / target.
(name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::INFO, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::INFO, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::INFO, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::INFO, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::INFO, {}, $($arg)+)
);

// Target / parent.
(target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::INFO, { $($field)* }, $($arg)*)
);
(target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::INFO, { $($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::INFO, { ?$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::INFO, { %$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::INFO, {}, $($arg)+)
);

// Name / parent.
(name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::INFO, { $($field)* }, $($arg)*)
);
(name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::INFO, { $($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::INFO, { ?$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::INFO, { %$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::INFO, {}, $($arg)+)
);

// Name.
(name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::INFO, { $($field)* }, $($arg)*)
);
(name: $name:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::INFO, { $($k).+ $($field)* })
);
(name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::INFO, { ?$($k).+ $($field)* })
);
(name: $name:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::INFO, { %$($k).+ $($field)* })
);
(name: $name:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, $crate::Level::INFO, {}, $($arg)+)
);

// Target.
(target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::INFO, { $($field)* }, $($arg)*)
);
(target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::INFO, { $($k).+ $($field)* })
);
(target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::INFO, { ?$($k).+ $($field)* })
);
(target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::INFO, { %$($k).+ $($field)* })
);
(target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, $crate::Level::INFO, {}, $($arg)+)
);

// Parent.
(parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { $($field)+ },
        $($arg)+
    )
);
(parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { $($k).+ = $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { ?$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { %$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { $($k).+, $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { ?$($k).+, $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        { %$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::INFO,
        {},
        $($arg)+
    )
);

// ...
({ $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { $($field)+ },
        $($arg)+
    )
);
($($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { $($k).+ = $($field)*}
    )
);
(?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { ?$($k).+ = $($field)*}
    )
);
(%$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { %$($k).+ = $($field)*}
    )
);
($($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { $($k).+, $($field)*}
    )
);
(?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { ?$($k).+, $($field)*}
    )
);
(%$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { %$($k).+, $($field)*}
    )
);
(?$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { ?$($k).+ }
    )
);
(%$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { %$($k).+ }
    )
);
($($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        { $($k).+ }
    )
);
($($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::INFO,
        $($arg)+
    )
);
}

#[macro_export]
macro_rules! warn {
// Name / target / parent.
(name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::WARN, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::WARN, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::WARN, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::WARN, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::WARN, {}, $($arg)+)
);

// Name / target.
(name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::WARN, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::WARN, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::WARN, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::WARN, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::WARN, {}, $($arg)+)
);

// Target / parent.
(target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::WARN, { $($field)* }, $($arg)*)
);
(target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::WARN, { $($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::WARN, { ?$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::WARN, { %$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::WARN, {}, $($arg)+)
);

// Name / parent.
(name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::WARN, { $($field)* }, $($arg)*)
);
(name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::WARN, { $($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::WARN, { ?$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::WARN, { %$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::WARN, {}, $($arg)+)
);

// Name.
(name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::WARN, { $($field)* }, $($arg)*)
);
(name: $name:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::WARN, { $($k).+ $($field)* })
);
(name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::WARN, { ?$($k).+ $($field)* })
);
(name: $name:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::WARN, { %$($k).+ $($field)* })
);
(name: $name:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, $crate::Level::WARN, {}, $($arg)+)
);

// Target.
(target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::WARN, { $($field)* }, $($arg)*)
);
(target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::WARN, { $($k).+ $($field)* })
);
(target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::WARN, { ?$($k).+ $($field)* })
);
(target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::WARN, { %$($k).+ $($field)* })
);
(target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, $crate::Level::WARN, {}, $($arg)+)
);

// Parent.
(parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { $($field)+ },
        $($arg)+
    )
);
(parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { $($k).+ = $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { ?$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { %$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { $($k).+, $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { ?$($k).+, $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        { %$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::WARN,
        {},
        $($arg)+
    )
);

// ...
({ $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { $($field)+ },
        $($arg)+
    )
);
($($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { $($k).+ = $($field)*}
    )
);
(?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { ?$($k).+ = $($field)*}
    )
);
(%$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { %$($k).+ = $($field)*}
    )
);
($($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { $($k).+, $($field)*}
    )
);
(?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { ?$($k).+, $($field)*}
    )
);
(%$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { %$($k).+, $($field)*}
    )
);
(?$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { ?$($k).+ }
    )
);
(%$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { %$($k).+ }
    )
);
($($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        { $($k).+ }
    )
);
($($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::WARN,
        $($arg)+
    )
);
}

#[macro_export]
macro_rules! error {
// Name / target / parent.
(name: $name:expr, target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::ERROR, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::ERROR, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::ERROR, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::ERROR, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, parent: $parent, $crate::Level::ERROR, {}, $($arg)+)
);

// Name / target.
(name: $name:expr, target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::ERROR, { $($field)* }, $($arg)*)
);
(name: $name:expr, target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::ERROR, { $($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::ERROR, { ?$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::ERROR, { %$($k).+ $($field)* })
);
(name: $name:expr, target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, target: $target, $crate::Level::ERROR, {}, $($arg)+)
);

// Target / parent.
(target: $target:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::ERROR, { $($field)* }, $($arg)*)
);
(target: $target:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::ERROR, { $($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::ERROR, { ?$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::ERROR, { %$($k).+ $($field)* })
);
(target: $target:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, parent: $parent, $crate::Level::ERROR, {}, $($arg)+)
);

// Name / parent.
(name: $name:expr, parent: $parent:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::ERROR, { $($field)* }, $($arg)*)
);
(name: $name:expr, parent: $parent:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::ERROR, { $($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::ERROR, { ?$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::ERROR, { %$($k).+ $($field)* })
);
(name: $name:expr, parent: $parent:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, parent: $parent, $crate::Level::ERROR, {}, $($arg)+)
);

// Name.
(name: $name:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::ERROR, { $($field)* }, $($arg)*)
);
(name: $name:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::ERROR, { $($k).+ $($field)* })
);
(name: $name:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::ERROR, { ?$($k).+ $($field)* })
);
(name: $name:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(name: $name, $crate::Level::ERROR, { %$($k).+ $($field)* })
);
(name: $name:expr, $($arg:tt)+ ) => (
    $crate::event!(name: $name, $crate::Level::ERROR, {}, $($arg)+)
);

// Target.
(target: $target:expr, { $($field:tt)* }, $($arg:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::ERROR, { $($field)* }, $($arg)*)
);
(target: $target:expr, $($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::ERROR, { $($k).+ $($field)* })
);
(target: $target:expr, ?$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::ERROR, { ?$($k).+ $($field)* })
);
(target: $target:expr, %$($k:ident).+ $($field:tt)* ) => (
    $crate::event!(target: $target, $crate::Level::ERROR, { %$($k).+ $($field)* })
);
(target: $target:expr, $($arg:tt)+ ) => (
    $crate::event!(target: $target, $crate::Level::ERROR, {}, $($arg)+)
);

// Parent.
(parent: $parent:expr, { $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { $($field)+ },
        $($arg)+
    )
);
(parent: $parent:expr, $($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { $($k).+ = $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { ?$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { %$($k).+ = $($field)*}
    )
);
(parent: $parent:expr, $($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { $($k).+, $($field)*}
    )
);
(parent: $parent:expr, ?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { ?$($k).+, $($field)*}
    )
);
(parent: $parent:expr, %$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        { %$($k).+, $($field)*}
    )
);
(parent: $parent:expr, $($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        parent: $parent,
        $crate::Level::ERROR,
        {},
        $($arg)+
    )
);

// ...
({ $($field:tt)+ }, $($arg:tt)+ ) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { $($field)+ },
        $($arg)+
    )
);
($($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { $($k).+ = $($field)*}
    )
);
(?$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { ?$($k).+ = $($field)*}
    )
);
(%$($k:ident).+ = $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { %$($k).+ = $($field)*}
    )
);
($($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { $($k).+, $($field)*}
    )
);
(?$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { ?$($k).+, $($field)*}
    )
);
(%$($k:ident).+, $($field:tt)*) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { %$($k).+, $($field)*}
    )
);
(?$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { ?$($k).+ }
    )
);
(%$($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { %$($k).+ }
    )
);
($($k:ident).+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        { $($k).+ }
    )
);
($($arg:tt)+) => (
    $crate::event!(
        target: module_path!(),
        $crate::Level::ERROR,
        $($arg)+
    )
);
}

#[doc(hidden)]
#[macro_export]
macro_rules! callsite {
(name: $name:expr, kind: $kind:expr, fields: $($fields:tt)*) => {{
    $crate::callsite! {
        name: $name,
        kind: $kind,
        target: module_path!(),
        level: $crate::Level::TRACE,
        fields: $($fields)*
    }
}};
(
    name: $name:expr,
    kind: $kind:expr,
    level: $lvl:expr,
    fields: $($fields:tt)*
) => {{
    $crate::callsite! {
        name: $name,
        kind: $kind,
        target: module_path!(),
        level: $lvl,
        fields: $($fields)*
    }
}};
(
    name: $name:expr,
    kind: $kind:expr,
    target: $target:expr,
    level: $lvl:expr,
    fields: $($fields:tt)*
) => {{
    static META: $crate::Metadata<'static> = {
        $crate::metadata! {
            name: $name,
            target: $target,
            level: $lvl,
            fields: $crate::fieldset!( $($fields)* ),
            callsite: &__CALLSITE,
            kind: $kind,
        }
    };
    static __CALLSITE: $crate::callsite::DefaultCallsite = $crate::callsite::DefaultCallsite::new(&META);
    __CALLSITE.register();
    &__CALLSITE
}};
}

#[doc(hidden)]
#[macro_export]
macro_rules! callsite2 {
(name: $name:expr, kind: $kind:expr, fields: $($fields:tt)*) => {{
    $crate::callsite2! {
        name: $name,
        kind: $kind,
        target: module_path!(),
        level: $crate::Level::TRACE,
        fields: $($fields)*
    }
}};
(
    name: $name:expr,
    kind: $kind:expr,
    level: $lvl:expr,
    fields: $($fields:tt)*
) => {{
    $crate::callsite2! {
        name: $name,
        kind: $kind,
        target: module_path!(),
        level: $lvl,
        fields: $($fields)*
    }
}};
(
    name: $name:expr,
    kind: $kind:expr,
    target: $target:expr,
    level: $lvl:expr,
    fields: $($fields:tt)*
) => {{
    static META: $crate::Metadata<'static> = {
        $crate::metadata! {
            name: $name,
            target: $target,
            level: $lvl,
            fields: $crate::fieldset!( $($fields)* ),
            callsite: &__CALLSITE,
            kind: $kind,
        }
    };
    $crate::callsite::DefaultCallsite::new(&META)
}};
}

#[macro_export]
#[doc(hidden)]
macro_rules! level_enabled {
($lvl:expr) => {
    $lvl <= $crate::level_filters::STATIC_MAX_LEVEL
        && $lvl <= $crate::level_filters::LevelFilter::current()
};
}

#[doc(hidden)]
#[macro_export]
macro_rules! valueset {

// === base case ===
(@ { $(,)* $($val:expr),* $(,)* }, $next:expr $(,)*) => {
    &[ $($val),* ]
};

(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+ = ?$val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&debug(&$val) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+ = %$val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&display(&$val) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+ = $val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&$val as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&$($k).+ as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, ?$($k:ident).+, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&debug(&$($k).+) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, %$($k:ident).+, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&display(&$($k).+) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+ = ?$val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&debug(&$val) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+ = %$val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&display(&$val) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+ = $val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&$val as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $($k:ident).+) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&$($k).+ as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, ?$($k:ident).+) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&debug(&$($k).+) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, %$($k:ident).+) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&display(&$($k).+) as &dyn Value)) },
        $next,
    )
};

// Handle literal names
(@ { $(,)* $($out:expr),* }, $next:expr, $k:literal = ?$val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&debug(&$val) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $k:literal = %$val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&display(&$val) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $k:literal = $val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&$val as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $k:literal = ?$val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&debug(&$val) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $k:literal = %$val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&display(&$val) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, $k:literal = $val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, $crate::__macro_support::Option::Some(&$val as &dyn Value)) },
        $next,
    )
};

// Handle constant names
(@ { $(,)* $($out:expr),* }, $next:expr, { $k:expr } = ?$val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, Some(&debug(&$val) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, { $k:expr } = %$val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, Some(&display(&$val) as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, { $k:expr } = $val:expr, $($rest:tt)*) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, Some(&$val as &dyn Value)) },
        $next,
        $($rest)*
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, { $k:expr } = ?$val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, Some(&debug(&$val) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, { $k:expr } = %$val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, Some(&display(&$val) as &dyn Value)) },
        $next,
    )
};
(@ { $(,)* $($out:expr),* }, $next:expr, { $k:expr } = $val:expr) => {
    $crate::valueset!(
        @ { $($out),*, (&$next, Some(&$val as &dyn Value)) },
        $next,
    )
};

(@ { $(,)* $($out:expr),* }, $next:expr, $($rest:tt)+) => {
    $crate::valueset!(@ { (&$next, $crate::__macro_support::Option::Some(&$crate::__macro_support::format_args!($($rest)+) as &dyn Value)), $($out),* }, $next, )
};

($fields:expr, $($kvs:tt)+) => {
    {
        #[allow(unused_imports)]
        use $crate::field::{debug, display, Value};
        let mut iter = $fields.iter();
        $fields.value_set($crate::valueset!(
            @ { },
            $crate::__macro_support::Iterator::next(&mut iter).expect("FieldSet corrupted (this is a bug)"),
            $($kvs)+
        ))
    }
};
($fields:expr,) => {
    {
        $fields.value_set(&[])
    }
};
}

#[doc(hidden)]
#[macro_export]
macro_rules! fieldset {
(@ { $(,)* $($out:expr),* $(,)* } $(,)*) => {
    &[ $($out),* ]
};

(@ { $(,)* $($out:expr),* } $($k:ident).+ = ?$val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $crate::__tracing_stringify!($($k).+) } $($rest)*)
};
(@ { $(,)* $($out:expr),* } $($k:ident).+ = %$val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $crate::__tracing_stringify!($($k).+) } $($rest)*)
};
(@ { $(,)* $($out:expr),* } $($k:ident).+ = $val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $crate::__tracing_stringify!($($k).+) } $($rest)*)
};
(@ { $(,)* $($out:expr),* } ?$($k:ident).+, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $crate::__tracing_stringify!($($k).+) } $($rest)*)
};
(@ { $(,)* $($out:expr),* } %$($k:ident).+, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $crate::__tracing_stringify!($($k).+) } $($rest)*)
};
(@ { $(,)* $($out:expr),* } $($k:ident).+, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $crate::__tracing_stringify!($($k).+) } $($rest)*)
};

// Handle literal names
(@ { $(,)* $($out:expr),* } $k:literal = ?$val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $k } $($rest)*)
};
(@ { $(,)* $($out:expr),* } $k:literal = %$val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $k } $($rest)*)
};
(@ { $(,)* $($out:expr),* } $k:literal = $val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $k } $($rest)*)
};

// Handle constant names
(@ { $(,)* $($out:expr),* } { $k:expr } = ?$val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $k } $($rest)*)
};
(@ { $(,)* $($out:expr),* } { $k:expr } = %$val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $k } $($rest)*)
};
(@ { $(,)* $($out:expr),* } { $k:expr } = $val:expr, $($rest:tt)*) => {
    $crate::fieldset!(@ { $($out),*, $k } $($rest)*)
};

(@ { $(,)* $($out:expr),* } $($rest:tt)+) => {
    $crate::fieldset!(@ { "message", $($out),*, })
};

($($args:tt)*) => {
    $crate::fieldset!(@ { } $($args)*,)
};

}

#[cfg(feature = "log")]
#[doc(hidden)]
#[macro_export]
macro_rules! level_to_log {
($level:expr) => {
    match $level {
        $crate::Level::ERROR => $crate::log::Level::Error,
        $crate::Level::WARN => $crate::log::Level::Warn,
        $crate::Level::INFO => $crate::log::Level::Info,
        $crate::Level::DEBUG => $crate::log::Level::Debug,
        _ => $crate::log::Level::Trace,
    }
};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __tracing_stringify {
($($t:tt)*) => {
    stringify!($($t)*)
};
}

#[cfg(not(feature = "log"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __tracing_log {
($level:expr, $callsite:expr, $value_set:expr) => {};
}

#[cfg(feature = "log")]
#[doc(hidden)]
#[macro_export]
macro_rules! __tracing_log {
($level:expr, $callsite:expr, $value_set:expr) => {
    $crate::if_log_enabled! { $level, {
        use $crate::log;
        let level = $crate::level_to_log!($level);
        if level <= log::max_level() {
            let meta = $callsite.metadata();
            let log_meta = log::Metadata::builder()
                .level(level)
                .target(meta.target())
                .build();
            let logger = log::logger();
            if logger.enabled(&log_meta) {
                $crate::__macro_support::__tracing_log(meta, logger, log_meta, $value_set)
            }
        }
    }}
};
}

#[cfg(not(feature = "log"))]
#[doc(hidden)]
#[macro_export]
macro_rules! if_log_enabled {
($lvl:expr, $e:expr;) => {
    $crate::if_log_enabled! { $lvl, $e }
};
($lvl:expr, $if_log:block) => {
    $crate::if_log_enabled! { $lvl, $if_log else {} }
};
($lvl:expr, $if_log:block else $else_block:block) => {
    $else_block
};
}

#[cfg(all(feature = "log", not(feature = "log-always")))]
#[doc(hidden)]
#[macro_export]
macro_rules! if_log_enabled {
($lvl:expr, $e:expr;) => {
    $crate::if_log_enabled! { $lvl, $e }
};
($lvl:expr, $if_log:block) => {
    $crate::if_log_enabled! { $lvl, $if_log else {} }
};
($lvl:expr, $if_log:block else $else_block:block) => {
    if $crate::level_to_log!($lvl) <= $crate::log::STATIC_MAX_LEVEL {
        if !$crate::dispatcher::has_been_set() {
            $if_log
        } else {
            $else_block
        }
    } else {
        $else_block
    }
};
}

#[cfg(all(feature = "log", feature = "log-always"))]
#[doc(hidden)]
#[macro_export]
macro_rules! if_log_enabled {
($lvl:expr, $e:expr;) => {
    $crate::if_log_enabled! { $lvl, $e }
};
($lvl:expr, $if_log:block) => {
    $crate::if_log_enabled! { $lvl, $if_log else {} }
};
($lvl:expr, $if_log:block else $else_block:block) => {
    if $crate::level_to_log!($lvl) <= $crate::log::STATIC_MAX_LEVEL {
        #[allow(unused_braces)]
        $if_log
    } else {
        $else_block
    }
};
}

//- /lib.rs crate:ra_test_fixture deps:tracing
fn foo() {
tracing::error!();
}
    "#,
        &["E0432", "inactive-code", "unresolved-macro-call", "syntax-error", "macro-error"],
    );
}
