// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use utils::make_indent;

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum ListTactic {
    // One item per row.
    Vertical,
    // All items on one row.
    Horizontal,
    // Try Horizontal layout, if that fails then vertical
    HorizontalVertical,
    // Pack as many items as possible per row over (possibly) many rows.
    Mixed,
}

#[derive(Eq, PartialEq, Debug, Copy, Clone)]
pub enum SeparatorTactic {
    Always,
    Never,
    Vertical,
}

// TODO having some helpful ctors for ListFormatting would be nice.
pub struct ListFormatting<'a> {
    pub tactic: ListTactic,
    pub separator: &'a str,
    pub trailing_separator: SeparatorTactic,
    pub indent: usize,
    // Available width if we layout horizontally.
    pub h_width: usize,
    // Available width if we layout vertically
    pub v_width: usize,
}

// Format a list of strings into a string.
// Precondition: all strings in items are trimmed.
pub fn write_list<'b>(items: &[(String, String)], formatting: &ListFormatting<'b>) -> String {
    if items.len() == 0 {
        return String::new();
    }

    let mut tactic = formatting.tactic;

    // Conservatively overestimates because of the changing separator tactic.
    let sep_count = if formatting.trailing_separator != SeparatorTactic::Never {
        items.len()
    } else {
        items.len() - 1
    };
    let sep_len = formatting.separator.len();
    let total_sep_len = (sep_len + 1) * sep_count;

    let total_width = calculate_width(items);

    // Check if we need to fallback from horizontal listing, if possible.
    if tactic == ListTactic::HorizontalVertical {
        if total_width + total_sep_len > formatting.h_width {
            tactic = ListTactic::Vertical;
        } else {
            tactic = ListTactic::Horizontal;
        }
    }

    // Now that we know how we will layout, we can decide for sure if there
    // will be a trailing separator.
    let trailing_separator = needs_trailing_separator(formatting.trailing_separator, tactic);

    // Create a buffer for the result.
    // TODO could use a StringBuffer or rope for this
    let alloc_width = if tactic == ListTactic::Horizontal {
        total_width + total_sep_len
    } else {
        total_width + items.len() * (formatting.indent + 1)
    };
    let mut result = String::with_capacity(alloc_width);

    let mut line_len = 0;
    let indent_str = &make_indent(formatting.indent);
    for (i, &(ref item, ref comment)) in items.iter().enumerate() {
        let first = i == 0;
        let separate = i != items.len() - 1 || trailing_separator;

        match tactic {
            ListTactic::Horizontal if !first => {
                result.push(' ');
            }
            ListTactic::Vertical if !first => {
                result.push('\n');
                result.push_str(indent_str);
            }
            ListTactic::Mixed => {
                let mut item_width = item.len();
                if separate {
                    item_width += sep_len;
                }

                if line_len > 0 && line_len + item_width > formatting.v_width {
                    result.push('\n');
                    result.push_str(indent_str);
                    line_len = 0;
                }

                if line_len > 0 {
                    result.push(' ');
                    line_len += 1;
                }

                line_len += item_width;
            }
            _ => {}
        }

        result.push_str(item);

        if tactic != ListTactic::Vertical && comment.len() > 0 {
            result.push(' ');
            result.push_str(comment);
        }

        if separate {
            result.push_str(formatting.separator);
        }

        if tactic == ListTactic::Vertical && comment.len() > 0 {
            result.push(' ');
            result.push_str(comment);
        }
    }

    result
}

fn needs_trailing_separator(separator_tactic: SeparatorTactic, list_tactic: ListTactic) -> bool {
    match separator_tactic {
        SeparatorTactic::Always => true,
        SeparatorTactic::Vertical => list_tactic == ListTactic::Vertical,
        SeparatorTactic::Never => false,
    }
}

fn calculate_width(items:&[(String, String)]) -> usize {
    let missed_width = items.iter().map(|&(_, ref s)| {
        let text_len = s.trim().len();
        if text_len > 0 {
            // We'll put a space before any comment.
            text_len + 1
        } else {
            text_len
        }
    }).fold(0, |a, l| a + l);
    let item_width = items.iter().map(|&(ref s, _)| s.len()).fold(0, |a, l| a + l);
    missed_width + item_width
}
