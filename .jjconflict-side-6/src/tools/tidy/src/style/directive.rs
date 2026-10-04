use std::cell::Cell;
use std::mem;
use std::path::Path;
use std::rc::Rc;

use crate::diagnostics::RunningCheck;

macro_rules! configurable_checks {
        (@parse_error_message doc = $doc:literal) => {
            Some($doc)
        };
        (@parse_error_message dont_check_unused) => {
            None
        };

        ($(#[$($doc: tt)*] $field: ident => $name: expr),* $(,)?) => {
        #[derive(Debug)]
        pub struct Directives {
            $(
                pub $field: NamedDirective
            ),*
        }

        impl Default for Directives {
            fn default() -> Self {
                Self {
                    $($field: NamedDirective {
                        directive: Default::default(),
                        name: $name,
                        error_message: configurable_checks!(@parse_error_message $($doc)*),
                    }),*
                }
            }
        }

        impl Directives {
            pub fn iter(&self) -> impl Iterator<Item = &NamedDirective> {
                vec![
                    $(
                        &self.$field
                    ),*
                ].into_iter()
            }

            pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut NamedDirective> {
                vec![
                    $(
                        &mut self.$field
                    ),*
                ].into_iter()
            }

            pub fn create_child(&self, new: Directives, check: &mut RunningCheck, file: &Path) -> Directives {
                Directives {
                    $($field: self.$field.create_child(new.$field, check, file)),*
                }
            }
        }
    };
}

const LINELENGTH_CHECK: &str = "linelength";
configurable_checks!(

    // We deliberately do not warn about this being unnecessary,
    // that would just lead to annoying churn.
    #[dont_check_unused]
    linelength => LINELENGTH_CHECK,

    // We deliberately do not warn about this being unnecessary,
    // that would just lead to annoying churn.
    #[dont_check_unused]
    filelength => "filelength",

    // ignore tidy for this entire file
    #[dont_check_unused]
    all => "all",

    /// ignoring CR characters unnecessarily
    cr => "cr",
    /// ignoring undocumented unsafe unnecessarily
    undocumented_unsafe => "undocumented-unsafe",
    /// ignoring tab characters unnecessarily
    tab => "tab",
    /// ignoring trailing whitespace unnecessarily
    end_whitespace => "end-whitespace",
    /// ignoring trailing newlines unnecessarily
    trailing_newlines => "trailing-newlines",
    /// ignoring leading newlines unnecessarily
    leading_newlines => "leading-newlines",
    /// ignoring leading newlines unnecessarily
    copyright => "copyright",
    /// ignoring dbg usage unnecessarily
    dbg => "dbg",
    /// ignoring odd backticks unnecessarily
    odd_backticks => "odd-backticks",
    /// ignoring todo usage unnecessarily
    todo => "todo",
);

#[derive(Debug)]
pub struct NamedDirective {
    directive: Directive,
    name: &'static str,
    error_message: Option<&'static str>,
}

impl NamedDirective {
    pub fn check_usage(&self, check: &mut RunningCheck, file: &Path) {
        let Some(message) = self.error_message else {
            self.force_discard_unsused_ignore();
            return;
        };

        if let Err(line) = self.directive.is_ignore_unused_and_mark_checked() {
            match line {
                LineNumber::Line(line) => {
                    check.error(format!("{}:{}: {}", file.display(), line, message));
                }
                LineNumber::WholeFile => {
                    check.error(format!("{}: {}", file.display(), message));
                }
            }
        }
    }

    fn create_child(&self, new: Self, check: &mut RunningCheck, file: &Path) -> Self {
        let directive = match (&self.directive, &new.directive) {
            // If both are deny, we don't care.
            (Directive::Deny, Directive::Deny) => Directive::Deny,
            // If (for example) ignored at the file level, but denied at the line level,
            // keep the ignore as a derived. Deny is the default, so we care about the
            // ignore.
            (Directive::Ignore { used, line_number, inherited: _ }, Directive::Deny) => {
                Directive::Ignore {
                    used: Rc::clone(used),
                    line_number: *line_number,
                    inherited: true,
                }
            }
            // If (for example) ignored at the line level, but not at the file level,
            // copy in the line-level one verbatim.
            (Directive::Deny, Directive::Ignore { .. }) => return new,
            // If (for example) ignored at the file level, and also at the line level,
            // keep the file-level one. It takes precedence. If a lint is ignored at
            // a file level, the line-level ignore should be marked "unused".
            (Directive::Ignore { used, line_number, inherited: _ }, Directive::Ignore { .. }) => {
                new.check_usage(check, file);
                Directive::Ignore {
                    used: Rc::clone(used),
                    line_number: *line_number,
                    inherited: true,
                }
            }
        };

        Self { directive, name: self.name, error_message: self.error_message }
    }

    pub fn is_ignore_and_defuse(&self) -> bool {
        if let Directive::Ignore { used, .. } = &self.directive {
            used.set(DirectiveUsed::Checked);
            true
        } else {
            false
        }
    }

    pub fn take(&mut self) -> Self {
        mem::replace(
            self,
            Self {
                directive: Default::default(),
                name: self.name,
                error_message: self.error_message,
            },
        )
    }

    /// Check whether we should error on this directive, or whether it was ignored.
    pub fn check(&self) -> Result<(), ()> {
        match &self.directive {
            Directive::Deny => Err(()),
            Directive::Ignore { used, .. } => {
                used.set(DirectiveUsed::Yes);
                Ok(())
            }
        }
    }

    /// Explicitly discard the fact that this directive may be ignored unnecessary.
    pub fn force_discard_unsused_ignore(&self) {
        self.is_ignore_and_defuse();
    }
}

impl Drop for NamedDirective {
    fn drop(&mut self) {
        if let Directive::Ignore { used, inherited: false, .. } = &self.directive
            && !matches!(used.get(), DirectiveUsed::Checked)
        {
            panic!("unchecked directive {} (call Directives::check_usage)", self.name);
        }
    }
}

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
pub enum Directive {
    /// By default, tidy always warns against style issues.
    #[default]
    Deny,

    /// `Ignore {used: No}` means that an `ignore-tidy-*` directive has been provided,
    /// but is unnecessary. `Ignore {used: Yes}` means that it is necessary
    /// (i.e. a warning would be produced if `ignore-tidy-*` was not present).
    Ignore {
        /// line on which the ignore was found
        line_number: LineNumber,
        used: Rc<Cell<DirectiveUsed>>,
        /// If this ignore is inherited from some higher-level,
        /// like from a file-level ignore, we only want to check whether
        /// it was used or not (and run the drop bomb) at the end of the file.
        inherited: bool,
    },
}

impl Directive {
    pub fn set_ignore(&mut self, line_number: LineNumber) {
        if let Self::Ignore { used, .. } = self {
            used.set(DirectiveUsed::No);
        } else {
            *self = Directive::Ignore {
                used: Rc::new(Cell::new(DirectiveUsed::No)),
                line_number,
                inherited: false,
            }
        }
    }

    /// Check whether this directive was ignored unnecessary.
    fn is_ignore_unused_and_mark_checked(&self) -> Result<(), LineNumber> {
        // only if inherted = false, otherwise, we shouldn't set checked and shouldn't care about its value
        if let Self::Ignore { used, line_number, inherited: false } = self {
            let used = used.replace(DirectiveUsed::Checked);
            if matches!(used, DirectiveUsed::No) { Err(*line_number) } else { Ok(()) }
        } else {
            Ok(())
        }
    }
}

impl Directives {
    pub fn check_usage(self, check: &mut RunningCheck, file: &Path) {
        for i in self.iter() {
            i.check_usage(check, file);
        }
    }

    pub fn from_str(
        path_str: &str,
        line_number: LineNumber,
        can_contain_directive_fastpath: bool,
        contents: &str,
    ) -> Self {
        let mut res = if !can_contain_directive_fastpath {
            Default::default()
        } else {
            Self::parse(line_number, contents)
        };

        // The rustdoc-json test syntax often requires very long lines, so the checks
        // for long lines aren't really useful.
        let always_ignore_linelength = path_str.contains("rustdoc-json");

        if always_ignore_linelength {
            res.linelength.directive.set_ignore(line_number);
        }

        res
    }

    pub fn parse(line_number: LineNumber, contents: &str) -> Self {
        let mut directives = Self::default();

        for directive in directives.iter_mut() {
            if match_ignore(
                contents,
                matches!(line_number, LineNumber::WholeFile),
                Some(directive.name),
            ) {
                directive.directive.set_ignore(line_number);
            }
        }

        directives
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LineNumber {
    Line(usize),
    /// file ignores were used, which scans the whole file for patterns.
    /// We don't know exactly on which line it happened.
    /// FIXME: do know
    WholeFile,
}

pub fn match_ignore(contents: &str, whole_file: bool, check: Option<&str>) -> bool {
    let comments = [("// ", ""), ("# ", ""), ("/* ", " */"), ("<!-- ", " -->")];
    let base = "ignore-tidy";
    let file = if whole_file { "file-" } else { "" };

    for (start, end) in comments {
        if let Some(check) = check {
            if contents.contains(&format!("{start}{base}-{file}{check}{end}")) {
                return true;
            }
        } else {
            if contents.contains(&format!("{start}{base}")) {
                return true;
            }
        }
    }

    false
}
