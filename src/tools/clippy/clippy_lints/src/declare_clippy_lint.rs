#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! declare_clippy_lint {
    (@
        $(#[doc = $lit:literal])*
        pub $lint_name:ident,
        $category:ident,
        $lintcategory:expr,
        $desc:literal,
        $version_expr:expr,
        $version_lit:literal
    ) => {
        rustc_session::declare_tool_lint! {
            $(#[doc = $lit])*
            #[clippy::version = $version_lit]
            pub clippy::$lint_name,
            $category,
            $desc,
            report_in_external_macro:true
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
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Restriction, $desc,
            Some($version), $version
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        style,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Style, $desc,
            Some($version), $version

        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        correctness,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Deny, crate::LintCategory::Correctness, $desc,
            Some($version), $version

        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        perf,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Perf, $desc,
            Some($version), $version
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        complexity,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Complexity, $desc,
            Some($version), $version
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        suspicious,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Warn, crate::LintCategory::Suspicious, $desc,
            Some($version), $version
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        nursery,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Nursery, $desc,
            Some($version), $version
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        pedantic,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Pedantic, $desc,
            Some($version), $version
        }
    };
    (
        $(#[doc = $lit:literal])*
        #[clippy::version = $version:literal]
        pub $lint_name:ident,
        cargo,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Cargo, $desc,
            Some($version), $version
        }
    };

    (
        $(#[doc = $lit:literal])*
        pub $lint_name:ident,
        internal,
        $desc:literal
    ) => {
        declare_clippy_lint! {@
            $(#[doc = $lit])*
            pub $lint_name, Allow, crate::LintCategory::Internal, $desc,
            None, "0.0.0"
        }
    };
}
