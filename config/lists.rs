// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Configuration options related to rewriting a list.

use config::config_type::ConfigType;
use config::IndentStyle;

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
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum ListTactic {
    // One item per row.
    Vertical,
    // All items on one row.
    Horizontal,
    // Try Horizontal layout, if that fails then vertical.
    HorizontalVertical,
    // HorizontalVertical with a soft limit of n characters.
    LimitedHorizontalVertical(usize),
    // Pack as many items as possible per row over (possibly) many rows.
    Mixed,
}

impl_enum_serialize_and_deserialize!(ListTactic, Vertical, Horizontal, HorizontalVertical, Mixed);

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SeparatorTactic {
    Always,
    Never,
    Vertical,
}

impl_enum_serialize_and_deserialize!(SeparatorTactic, Always, Never, Vertical);

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
#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SeparatorPlace {
    Front,
    Back,
}

impl_enum_serialize_and_deserialize!(SeparatorPlace, Front, Back);

impl SeparatorPlace {
    pub fn is_front(&self) -> bool {
        *self == SeparatorPlace::Front
    }

    pub fn is_back(&self) -> bool {
        *self == SeparatorPlace::Back
    }

    pub fn from_tactic(
        default: SeparatorPlace,
        tactic: DefinitiveListTactic,
        sep: &str,
    ) -> SeparatorPlace {
        match tactic {
            DefinitiveListTactic::Vertical => default,
            _ => if sep == "," {
                SeparatorPlace::Back
            } else {
                default
            },
        }
    }
}
