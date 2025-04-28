#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! declare_clippy_lint {
    (@
        $(#[doc = $lit:literal])*
        pub $lint_name:ident,
        $level:ident,
        $lintcategory:expr,
        $desc:literal,
        $version_expr:expr,
        $version_lit:literal
        $(, $eval_always: literal)?
    ) => {
        rustc_session::declare_tool_lint! {
            $(#[doc = $lit])*
            #[clippy::version = $version_lit]
            pub clippy::$lint_name,
            $level,
            $desc,
            report_in_external_macro:true
            $(, @eval_always = $eval_always)?
        }

        pub(crate) static ${concat($lint_name, _INFO)}: &'static crate::LintInfo = &crate::LintInfo {
            lint: &$lint_name,
            category:  $lintcategory,
            explanation: concat!($($lit,"\n",)*),
            location: concat!(file!(), "#L", line!()),
            version: $version_expr
        };
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        restriction,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Restriction, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        style,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Style, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        correctness,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Deny, crate::LintCategory::Correctness, $desc,
            Some($version), $version
            $(, $eval_always)?

        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        perf,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Perf, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        complexity,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Complexity, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        suspicious,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Suspicious, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        nursery,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Nursery, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        pedantic,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Pedantic, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        cargo,
        $desc:literal
        $(, @eval_always = $eval_always: literal)?
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Cargo, $desc,
            Some($version), $version
            $(, $eval_always)?
        }
    };
}
