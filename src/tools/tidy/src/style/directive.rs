use std::cell::Cell;
use std::path::Path;

use crate::diagnostics::RunningCheck;

const LINELENGTH_CHECK: &str = "linelength";

macro_rules! configurable_checks {
        ($($name: ident => $field: expr),* $(,)?) => {
        #[derive(Debug, Default)]
        pub struct Directives<'a> {
            $(
                pub $name: Directive<'a>
            ),*
        }

        impl<'a> Directives<'a> {
            pub fn iter(&self) -> impl Iterator<Item = (&'static str, &Directive<'a>)> {
                vec![
                    $(
                        ($field, &self.$name)
                    ),*
                ].into_iter()
            }

            pub fn iter_mut(&mut self) -> impl Iterator<Item = (&'static str, &mut Directive<'a>)> {
                vec![
                    $(
                        ($field, &mut self.$name)
                    ),*
                ].into_iter()
            }

            pub fn create_child<'x>(&'x self, new: Directives<'x>) -> Directives<'x> {
                Directives {
                    $($name: self.$name.create_child(new.$name)),*
                }
            }
        }
    };
}

configurable_checks!(
    cr => "cr",
    undocumented_unsafe => "undocumented-unsafe",
    tab => "tab",
    linelength => LINELENGTH_CHECK,
    filelength => "filelength",
    end_whitespace => "end-whitespace",
    trailing_newlines => "trailing-newlines",
    leading_newlines => "leading-newlines",
    copyright => "copyright",
    dbg => "dbg",
    odd_backticks => "odd-backticks",
    todo => "todo",
);

/// When a directive is set to "ignore",  it means a normally-active
/// tidy lint is now ignored. This can be bad, if it's ignored for no
/// reason, so we track whether the ignore is actualy ignoring something.
#[derive(Clone, Copy, Debug)]
pub enum DirectiveUsed {
    /// A lint is ignored, and something used that fact.
    /// i.e. A lint would've been emitted if it wasn't ignored.
    Yes,
    /// A lint is ignored but nothing yet used this ignore.
    No,
    /// For drop-bomb behavior: a directive is ignored, and we've
    /// checked whether it was used or not. Directives panic on drop
    /// if they aren't checked, to ensure we always emit a lint when
    /// a directive is uselessly ignored
    Checked,
}

#[derive(Debug, Default)]
pub enum Directive<'a> {
    /// By default, tidy always warns against style issues.
    #[default]
    Deny,

    /// `Ignore {used: No}` means that an `ignore-tidy-*` directive has been provided,
    /// but is unnecessary. `Ignore {used: Yes}` means that it is necessary
    /// (i.e. a warning would be produced if `ignore-tidy-*` was not present).
    Ignore { used: Cell<DirectiveUsed> },

    /// Some higher-level rule decides whether this directive is denied or ignored.
    Derived(&'a Directive<'a>),
}

impl<'a> Drop for Directive<'a> {
    fn drop(&mut self) {
        if let Self::Ignore { used } = self
            && !matches!(used.get(), DirectiveUsed::Checked)
        {
            panic!("unchecked directive (call Directives::check_usage)");
        }
    }
}

impl<'a> Directive<'a> {
    pub fn set_ignore(&mut self) {
        if let Self::Ignore { used } = self {
            used.set(DirectiveUsed::No);
        } else {
            *self = Directive::Ignore { used: Cell::new(DirectiveUsed::No) }
        }
    }

    /// Check whether this directive was ignored unnecessary.
    fn is_ignore_unused(&self) -> bool {
        if let Self::Ignore { used } = self {
            let used = used.replace(DirectiveUsed::Checked);
            matches!(used, DirectiveUsed::No)
        } else {
            false
        }
    }

    pub fn is_ignore_and_defuse(&self) -> bool {
        if let Self::Ignore { used } = self {
            used.set(DirectiveUsed::Checked);
            true
        } else {
            false
        }
    }

    /// Explicitly discard the fact that this directive may be ignored unnecessary.
    fn force_discard_unsused_ignore(&self) {
        self.is_ignore_and_defuse();
    }

    pub fn create_child<'x>(&'x self, new: Directive<'x>) -> Directive<'x> {
        match (self, new) {
            // If both are deny, we don't care.
            (Directive::Deny, Directive::Deny) => Directive::Deny,
            // If (for example) ignored at the file level, but denied at the line level,
            // keep the ignore as a derived. Deny is the default, so we care about the
            // ignore.
            (Directive::Ignore { .. }, Directive::Deny) => Directive::Derived(self),
            // If (for example) ignored at the line level, but not at the file level,
            // copy in the line-level one verbatim.
            (Directive::Deny, new @ Directive::Ignore { .. }) => new,
            // If (for example) ignored at the file level, and also at the line level,
            // keep the line-level one. It takes precedence. If all lints are ignored
            // per-line, the file-level ignore should be marked "unused".
            (Directive::Ignore { .. }, new @ Directive::Ignore { .. }) => new,
            // If self is already derived, shorten the path.
            (Directive::Derived(parent), new) => parent.create_child(new),
            // What are you even doing.
            (_, Directive::Derived(_)) => {
                unimplemented!()
            }
        }
    }

    /// Check whether we should error on this directive, or whether it was ignored.
    pub fn check(&self) -> Result<(), ()> {
        match self {
            Self::Deny => Err(()),
            Self::Ignore { used } => {
                used.set(DirectiveUsed::Yes);
                Ok(())
            }
            Self::Derived(directive) => directive.check(),
        }
    }
}

impl<'a> Directives<'a> {
    pub fn check_usage(self, check: &mut RunningCheck, file: &Path) {
        let Self {
            cr,
            undocumented_unsafe,
            tab,
            end_whitespace,
            trailing_newlines,
            leading_newlines,
            copyright,
            dbg,
            odd_backticks,
            todo,

            linelength,
            filelength,
        } = self;

        if cr.is_ignore_unused() {
            check.error(format!("{}: ignoring CR characters unnecessarily", file.display()));
        }
        if tab.is_ignore_unused() {
            check.error(format!("{}: ignoring tab characters unnecessarily", file.display()));
        }
        if end_whitespace.is_ignore_unused() {
            check.error(format!("{}: ignoring trailing whitespace unnecessarily", file.display()));
        }
        if trailing_newlines.is_ignore_unused() {
            check.error(format!("{}: ignoring trailing newlines unnecessarily", file.display()));
        }
        if leading_newlines.is_ignore_unused() {
            check.error(format!("{}: ignoring leading newlines unnecessarily", file.display()));
        }
        if copyright.is_ignore_unused() {
            check.error(format!("{}: ignoring copyright unnecessarily", file.display()));
        }
        if todo.is_ignore_unused() {
            check.error(format!("{}: ignoring todo usage unnecessarily", file.display()));
        }
        if dbg.is_ignore_unused() {
            check.error(format!("{}: ignoring dbg usage unnecessarily", file.display()));
        }
        if odd_backticks.is_ignore_unused() {
            check.error(format!("{}: ignoring odd backticks unnecessarily", file.display()));
        }
        if undocumented_unsafe.is_ignore_unused() {
            check.error(format!("{}: ignoring undocumented unsafe unnecessarily", file.display()));
        }

        // We deliberately do not warn about these being unnecessary,
        // that would just lead to annoying churn.
        linelength.force_discard_unsused_ignore();
        filelength.force_discard_unsused_ignore();
    }

    // Use a fixed size array in the return type to catch mistakes with changing `CONFIGURABLE_CHECKS`
    // without changing the code in `check` easier.
    pub fn from_line(
        path_str: &str,
        can_contain_directive_fastpath: bool,
        whole_file: bool,
        contents: &str,
    ) -> Self {
        let mut directives = Self::default();

        // The rustdoc-json test syntax often requires very long lines, so the checks
        // for long lines aren't really useful.
        let always_ignore_linelength = path_str.contains("rustdoc-json");

        if !can_contain_directive_fastpath && !always_ignore_linelength {
            return directives;
        }

        for (check, directive) in directives.iter_mut() {
            if check == LINELENGTH_CHECK && always_ignore_linelength {
                directive.set_ignore();
            }

            if match_ignore(contents, whole_file, Some(check)) {
                directive.set_ignore();
            }
        }

        directives
    }
}

pub fn match_ignore(contents: &str, whole_file: bool, check: Option<&str>) -> bool {
    let check = check.unwrap_or("");
    let file = if whole_file { "file-" } else { "" };
    contents.contains(&format!("// ignore-tidy-{check}"))
        || contents.contains(&format!("# ignore-tidy-{file}{check}"))
        || contents.contains(&format!("/* ignore-tidy-{file}{check} */"))
        || contents.contains(&format!("<!-- ignore-tidy-{file}{check} -->"))
}
