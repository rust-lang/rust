#[macro_export]
macro_rules! register_diagnostic {
    ($code:tt, $description:tt) => (__register_diagnostic! { $code, $description });
    ($code:tt) => (__register_diagnostic! { $code })
}

#[macro_export]
macro_rules! span_fatal {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.span_fatal_with_code(
            $span,
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! span_err {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.span_err_with_code(
            $span,
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! span_warn {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.span_warn_with_code(
            $span,
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! struct_err {
    ($session:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.struct_err_with_code(
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! span_err_or_warn {
    ($is_warning:expr, $session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        if $is_warning {
            $session.span_warn_with_code(
                $span,
                &format!($($message)*),
                $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
            )
        } else {
            $session.span_err_with_code(
                $span,
                &format!($($message)*),
                $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
            )
        }
    })
}

#[macro_export]
macro_rules! struct_span_fatal {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.struct_span_fatal_with_code(
            $span,
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! struct_span_err {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.struct_span_err_with_code(
            $span,
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! stringify_error_code {
    ($code:ident) => ({
        __diagnostic_used!($code);
        $crate::errors::DiagnosticId::Error(stringify!($code).to_owned())
    })
}

#[macro_export]
macro_rules! type_error_struct {
    ($session:expr, $span:expr, $typ:expr, $code:ident, $($message:tt)*) => ({
        if $typ.references_error() {
            $session.diagnostic().struct_dummy()
        } else {
            struct_span_err!($session, $span, $code, $($message)*)
        }
    })
}

#[macro_export]
macro_rules! struct_span_warn {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        $session.struct_span_warn_with_code(
            $span,
            &format!($($message)*),
            $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
        )
    })
}

#[macro_export]
macro_rules! struct_span_err_or_warn {
    ($is_warning:expr, $session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        __diagnostic_used!($code);
        if $is_warning {
            $session.struct_span_warn_with_code(
                $span,
                &format!($($message)*),
                $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
            )
        } else {
            $session.struct_span_err_with_code(
                $span,
                &format!($($message)*),
                $crate::errors::DiagnosticId::Error(stringify!($code).to_owned()),
            )
        }
    })
}

#[macro_export]
macro_rules! span_note {
    ($err:expr, $span:expr, $($message:tt)*) => ({
        ($err).span_note($span, &format!($($message)*));
    })
}

#[macro_export]
macro_rules! span_help {
    ($err:expr, $span:expr, $($message:tt)*) => ({
        ($err).span_help($span, &format!($($message)*));
    })
}

#[macro_export]
macro_rules! help {
    ($err:expr, $($message:tt)*) => ({
        ($err).help(&format!($($message)*));
    })
}

#[macro_export]
macro_rules! register_diagnostics {
    ($($code:tt),*) => (
        $(register_diagnostic! { $code })*
    );
    ($($code:tt),*,) => (
        $(register_diagnostic! { $code })*
    )
}

#[macro_export]
macro_rules! register_long_diagnostics {
    ($($code:tt: $description:tt),*) => (
        $(register_diagnostic! { $code, $description })*
    );
    ($($code:tt: $description:tt),*,) => (
        $(register_diagnostic! { $code, $description })*
    )
}
