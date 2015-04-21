// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use make_indent;

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
pub fn write_list<'b>(items:&[(String, String)], formatting: &ListFormatting<'b>) -> String {
    if items.len() == 0 {
        return String::new();
    }

    let mut tactic = formatting.tactic;

    let h_width = formatting.h_width;
    let v_width = formatting.v_width;
    let sep_len = formatting.separator.len();

    // Conservatively overestimates because of the changing separator tactic.
    let sep_count = if formatting.trailing_separator != SeparatorTactic::Never {
        items.len()
    } else {
        items.len() - 1
    };

    // TODO count dead space too.
    let total_width = items.iter().map(|&(ref s, _)| s.len()).fold(0, |a, l| a + l);

    // Check if we need to fallback from horizontal listing, if possible.
    if tactic == ListTactic::HorizontalVertical { 
        if (total_width + (sep_len + 1) * sep_count) > h_width {
            tactic = ListTactic::Vertical;
        } else {
            tactic = ListTactic::Horizontal;
        }
    }

    // Now that we know how we will layout, we can decide for sure if there
    // will be a trailing separator.
    let trailing_separator = match formatting.trailing_separator {
        SeparatorTactic::Always => true,
        SeparatorTactic::Vertical => tactic == ListTactic::Vertical,
        SeparatorTactic::Never => false,
    };

    // Create a buffer for the result.
    // TODO could use a StringBuffer or rope for this
    let alloc_width = if tactic == ListTactic::Horizontal {
        total_width + (sep_len + 1) * sep_count
    } else {
        total_width + items.len() * (formatting.indent + 1)
    };
    let mut result = String::with_capacity(alloc_width);

    let mut line_len = 0;
    let indent_str = &make_indent(formatting.indent);
    for (i, &(ref item, _)) in items.iter().enumerate() {
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

                if line_len > 0 && line_len + item_width > v_width {
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
        
        if separate {
            result.push_str(formatting.separator);
        }
        // TODO dead spans
    }

    result
}
