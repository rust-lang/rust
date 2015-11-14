// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Indent;
use rewrite::{Rewrite, RewriteContext};
use utils::{wrap_str, format_mutability, span_after};
use lists::{format_item_list, itemize_list};
use expr::{rewrite_unary_prefix, rewrite_pair, rewrite_tuple};
use types::rewrite_path;

use syntax::ast::{PatWildKind, BindingMode, Pat, Pat_};

// FIXME(#18): implement pattern formatting.
impl Rewrite for Pat {
    fn rewrite(&self, context: &RewriteContext, width: usize, offset: Indent) -> Option<String> {
        match self.node {
            Pat_::PatBox(ref pat) => {
                rewrite_unary_prefix(context, "box ", &**pat, width, offset)
            }
            Pat_::PatIdent(binding_mode, ident, None) => {
                let (prefix, mutability) = match binding_mode {
                    BindingMode::BindByRef(mutability) => ("ref ", mutability),
                    BindingMode::BindByValue(mutability) => ("", mutability),
                };
                let mut_infix = format_mutability(mutability);
                let result = format!("{}{}{}", prefix, mut_infix, ident.node);
                wrap_str(result, context.config.max_width, width, offset)
            }
            Pat_::PatWild(kind) => {
                let result = match kind {
                    PatWildKind::PatWildSingle => "_",
                    PatWildKind::PatWildMulti => "..",
                };
                if result.len() <= width {
                    Some(result.to_owned())
                } else {
                    None
                }
            }
            Pat_::PatQPath(ref q_self, ref path) => {
                rewrite_path(context, true, Some(q_self), path, width, offset)
            }
            Pat_::PatRange(ref lhs, ref rhs) => {
                rewrite_pair(&**lhs, &**rhs, "", "...", "", context, width, offset)
            }
            Pat_::PatRegion(ref pat, mutability) => {
                let prefix = format!("&{}", format_mutability(mutability));
                rewrite_unary_prefix(context, &prefix, &**pat, width, offset)
            }
            Pat_::PatTup(ref items) => {
                rewrite_tuple(context, items, self.span, width, offset)
            }
            Pat_::PatEnum(ref path, Some(ref pat_vec)) => {
                let path_str = try_opt!(::types::rewrite_path(context,
                                                              true,
                                                              None,
                                                              path,
                                                              width,
                                                              offset));

                if pat_vec.is_empty() {
                    Some(path_str)
                } else {
                    // 1 = (
                    let width = try_opt!(width.checked_sub(path_str.len() + 1));
                    let offset = offset + path_str.len() + 1;
                    let items = itemize_list(context.codemap,
                                             pat_vec.iter(),
                                             ")",
                                             |item| item.span.lo,
                                             |item| item.span.hi,
                                             |item| item.rewrite(context, width, offset),
                                             span_after(self.span, "(", context.codemap),
                                             self.span.hi);
                    Some(format!("{}({})",
                                 path_str,
                                 try_opt!(format_item_list(items, width, offset, context.config))))
                }
            }
            Pat_::PatLit(ref expr) => expr.rewrite(context, width, offset),
            // FIXME(#8): format remaining pattern variants.
            Pat_::PatIdent(_, _, Some(..)) |
            Pat_::PatEnum(_, None) |
            Pat_::PatStruct(..) |
            Pat_::PatVec(..) |
            Pat_::PatMac(..) => {
                wrap_str(context.snippet(self.span),
                         context.config.max_width,
                         width,
                         offset)
            }
        }
    }
}
