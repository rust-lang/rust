//! Configuration options related to rewriting a list.

use rustfmt_config_proc_macro::config_type;

use crate::config::IndentStyle;

/// The definitive formatting tactic for lists.
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum DefinitiveListTactic {
    Vertical,
    Horizontal,
    Mixed,
    /// Special case tactic for `format!()`, `write!()` style macros.
    SpecialMacro(usize),
}

impl DefinitiveListTactic {
    pub fn ends_with_newline(&self, indent_style: IndentStyle) -> bool {
        match indent_style {
            IndentStyle::Block => *self != DefinitiveListTactic::Horizontal,
            IndentStyle::Visual => false,
        }
    }
}

/// Formatting tactic for lists. This will be cast down to a
/// `DefinitiveListTactic` depending on the number and length of the items and
/// their comments.
#[config_type]
pub enum ListTactic {
    /// One item per row.
    Vertical,
    /// All items on one row.
    Horizontal,
    /// Try Horizontal layout, if that fails then vertical.
    HorizontalVertical,
    /// HorizontalVertical with a soft limit of n characters.
    LimitedHorizontalVertical(usize),
    /// Pack as many items as possible per row over (possibly) many rows.
    Mixed,
}

#[config_type]
pub enum SeparatorTactic {
    Always,
    Never,
    Vertical,
}

impl SeparatorTactic {
    pub fn from_bool(b: bool) -> SeparatorTactic {
        if b {
            SeparatorTactic::Always
        } else {
            SeparatorTactic::Never
        }
    }
}

/// Where to put separator.
#[config_type]
pub enum SeparatorPlace {
    Front,
    Back,
}

impl SeparatorPlace {
    pub fn is_front(self) -> bool {
        self == SeparatorPlace::Front
    }

    pub fn is_back(self) -> bool {
        self == SeparatorPlace::Back
    }

    pub fn from_tactic(
        default: SeparatorPlace,
        tactic: DefinitiveListTactic,
        sep: &str,
    ) -> SeparatorPlace {
        match tactic {
            DefinitiveListTactic::Vertical => default,
            _ => {
                if sep == "," {
                    SeparatorPlace::Back
                } else {
                    default
                }
            }
        }
    }
}
