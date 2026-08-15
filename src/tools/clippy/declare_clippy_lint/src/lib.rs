#![feature(macro_metavar_expr_concat, rustc_private)]

extern crate rustc_lint;

use rustc_lint::{Lint, LintId, LintStore};

// Needed by `declare_clippy_lint!`.
pub extern crate rustc_session;

#[derive(Default)]
pub struct LintListBuilder {
    lints: Vec<&'static Lint>,
    all: Vec<LintId>,
    cargo: Vec<LintId>,
    complexity: Vec<LintId>,
    correctness: Vec<LintId>,
    nursery: Vec<LintId>,
    pedantic: Vec<LintId>,
    perf: Vec<LintId>,
    restriction: Vec<LintId>,
    style: Vec<LintId>,
    suspicious: Vec<LintId>,
}
impl LintListBuilder {
    pub fn insert(&mut self, lints: &[&LintInfo]) {
        #[allow(clippy::enum_glob_use)]
        use LintCategory::*;

        self.lints.extend(lints.iter().map(|&x| x.lint));
        for &&LintInfo { lint, category, .. } in lints {
            let (all, cat) = match category {
                Complexity => (Some(&mut self.all), &mut self.complexity),
                Correctness => (Some(&mut self.all), &mut self.correctness),
                Perf => (Some(&mut self.all), &mut self.perf),
                Style => (Some(&mut self.all), &mut self.style),
                Suspicious => (Some(&mut self.all), &mut self.suspicious),
                Cargo => (None, &mut self.cargo),
                Nursery => (None, &mut self.nursery),
                Pedantic => (None, &mut self.pedantic),
                Restriction => (None, &mut self.restriction),
            };
            if let Some(all) = all {
                all.push(LintId::of(lint));
            }
            cat.push(LintId::of(lint));
        }
    }

    pub fn register(self, store: &mut LintStore) {
        store.register_lints(&self.lints);
        store.register_group(true, "clippy::all", Some("clippy_all"), self.all);
        store.register_group(true, "clippy::cargo", Some("clippy_cargo"), self.cargo);
        store.register_group(true, "clippy::complexity", Some("clippy_complexity"), self.complexity);
        store.register_group(
            true,
            "clippy::correctness",
            Some("clippy_correctness"),
            self.correctness,
        );
        store.register_group(true, "clippy::nursery", Some("clippy_nursery"), self.nursery);
        store.register_group(true, "clippy::pedantic", Some("clippy_pedantic"), self.pedantic);
        store.register_group(true, "clippy::perf", Some("clippy_perf"), self.perf);
        store.register_group(
            true,
            "clippy::restriction",
            Some("clippy_restriction"),
            self.restriction,
        );
        store.register_group(true, "clippy::style", Some("clippy_style"), self.style);
        store.register_group(true, "clippy::suspicious", Some("clippy_suspicious"), self.suspicious);
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LintCategory {
    Cargo,
    Complexity,
    Correctness,
    Nursery,
    Pedantic,
    Perf,
    Restriction,
    Style,
    Suspicious,
}
impl LintCategory {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cargo => "cargo",
            Self::Complexity => "complexity",
            Self::Correctness => "correctness",
            Self::Nursery => "nursery",
            Self::Pedantic => "pedantic",
            Self::Perf => "perf",
            Self::Restriction => "restriction",
            Self::Style => "style",
            Self::Suspicious => "suspicious",
        }
    }
}

pub struct LintInfo {
    pub lint: &'static Lint,
    pub category: LintCategory,
    pub explanation: &'static str,
    /// e.g. `clippy_lints/src/absolute_paths.rs#43`
    pub location: &'static str,
    pub version: &'static str,
}

impl LintInfo {
    /// Returns the lint name in lowercase without the `clippy::` prefix
    #[must_use]
    #[expect(clippy::missing_panics_doc)]
    pub fn name_lower(&self) -> String {
        self.lint.name.strip_prefix("clippy::").unwrap().to_ascii_lowercase()
    }
}

#[macro_export]
macro_rules! declare_clippy_lint_inner {
    (
        $(#[doc = $docs:literal])*
        #[clippy::version = $version:literal]
        $vis:vis $lint_name:ident,
        $level:ident,
        $category:ident,
        $desc:literal
        $(, @eval_always = $eval_always:literal)?
    ) => {
        $crate::rustc_session::declare_tool_lint! {
            $(#[doc = $docs])*
            #[clippy::version = $version]
            $vis clippy::$lint_name,
            $level,
            $desc,
            report_in_external_macro:true
            $(, @eval_always = $eval_always)?
        }

        pub(crate) static ${concat($lint_name, _INFO)}: &'static $crate::LintInfo = &$crate::LintInfo {
            lint: $lint_name,
            category: $crate::LintCategory::$category,
            explanation: concat!($($docs,"\n",)*),
            location: concat!(file!(), "#L", line!()),
            version: $version,
        };
    };
}

#[macro_export]
macro_rules! declare_clippy_lint {
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        correctness,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Deny,
            Correctness,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        complexity,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Warn,
            Complexity,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        perf,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Warn,
            Perf,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        style,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Warn,
            Style,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        suspicious,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Warn,
            Suspicious,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        cargo,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Allow,
            Cargo,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        nursery,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Allow,
            Nursery,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        pedantic,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Allow,
            Pedantic,
            $($rest)*
        }
    };
    (
        $(#[$($meta:tt)*])*
        $vis:vis $lint_name:ident,
        restriction,
        $($rest:tt)*
    ) => {
        $crate::declare_clippy_lint_inner! {
            $(#[$($meta)*])*
            $vis $lint_name,
            Allow,
            Restriction,
            $($rest)*
        }
    };
}
