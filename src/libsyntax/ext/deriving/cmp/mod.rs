// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ast::{ident, expr};
use codemap::span;
use ast;
use ext::base::ExtCtxt;
use ext::deriving::generic::*;
use ext::deriving::{DerivingOptions, NoOptions, Lit, List};

use std::vec;

pub mod eq;
pub mod totaleq;
pub mod ord;
pub mod totalord;

pub struct CmpOptions_ {
    /// Ensure that we don't show errors several times.
    shown_errors: bool,
    test_order: ~[(ident, span)],
    ignore: ~[(ident, span)],
    reverse: ~[(ident, span)]
}
pub struct CmpOptions(Option<CmpOptions_>);

impl CmpOptions {
    pub fn parse(cx: @ExtCtxt, _span: span, trait_name: &str,
                 options: DerivingOptions,
                 allow_ignore: bool, allow_reverse: bool) -> Option<CmpOptions> {
        // Trait(<option>(...))
        match options {
            NoOptions => Some(CmpOptions(None)),
            Lit(ref lit) => {
                cx.span_err(lit.span,
                            fmt!("`#[deriving(%s)]` does not \
                                  accept options with this syntax",
                                 trait_name));
                None
            }
            List(ref list) => {
                let mut cmp_options = CmpOptions_ {
                    shown_errors: false,
                    test_order: ~[],
                    ignore: ~[],
                    reverse: ~[]
                };

                let mut no_errors = true;
                for item in list.iter() {
                    match item.node {
                        ast::MetaList(name, ref fields) => {
                            no_errors = no_errors &&
                                cmp_options.parse_inner(cx,
                                                        trait_name, allow_ignore, allow_reverse,
                                                        item.span, name,
                                                        *fields)
                        }
                        _ => {
                            cx.span_err(
                                item.span,
                                fmt!("`#[deriving(%s)]` only accepts %s", trait_name,
                                     format_allowed(allow_ignore, allow_reverse, "(...)")));
                            no_errors = false;
                        }
                    }
                }
                if no_errors {
                    Some(CmpOptions(Some(cmp_options)))
                } else {
                    None
                }
            }
        }
    }

    /// Run a normal CombineSubstructureFunc with reordered
    /// fields. Use `first_run` to control whether error messages that
    /// occur on every run should be shown or not.
    pub fn call_substructure(&mut self, cx: @ExtCtxt, span: span,
                             substr: &Substructure,
                             f: CombineSubstructureFunc) -> @expr {
        match self.process_substructure(cx, span, substr.fields) {
            Left(ref new_substr_fields) => {
                let new_substr = Substructure {
                    fields: new_substr_fields,
                    .. *substr
                };
                f(cx, span, &new_substr)
            }
            Right(true) => f(cx, span, substr), // no changes, use the old one

            // whatever, just fill in something to get this to
            // function to work; we should never actually run this
            // expression (because `None` here means an error has
            // occurred already) so it'll never occur in the final,
            // compiled code.
            Right(false) => cx.expr_unreachable(span)
        }
    }

    /// Reorders the fields of `Struct(~[Some(id), self, ~[other],
    /// ...])` and returns the new fields as `Left(fields)`, and if no
    /// changes were necessary returns `Right(true)`, if an error
    /// occurred (e.g. options on a enum or tuple struct) returns
    /// `Right(false)`.
    fn process_substructure(&mut self,
                            cx: @ExtCtxt,
                            span: span,
                            fields: &SubstructureFields)
        -> Either<SubstructureFields<'static>, bool> {
        // say no to rightward drift.
        let options = match **self {
            None => return Right(true), // no options, no adjustments.
            Some(ref mut options) => options
        };

        // extract the list of fields
        let fields_list = match *fields {
            // The only valid case: a struct with named fields, and
            // the method has precisely 2 Self args.
            Struct(ref list) => {
                // should be [(Some(id), self, [other]), ...]
                if list.is_empty() || list[0].n0_ref().is_none() {
                    // Some(id) is actually None, or there are no fields
                    do options.oneshot_error {
                        cx.span_err(span,
                                    fmt!("cannot use options on a %s struct",
                                         if list.is_empty() {"unit"} else {"tuple"}));
                    }
                    return Right(false);
                } else if list[0].n2_ref().len() != 1 { // explicit self is separate
                    cx.span_bug(span,
                                fmt!("CmpOptions used for a method with %u Self \
                                      args (should be 2)",
                                     list[0].n2_ref().len() + 1)); // account for explicit self
                }

                list.as_slice()
            }
            EnumMatching(*) | EnumNonMatching(*) => {
                do options.oneshot_error {
                    cx.span_err(span, "cannot use options on an enum");
                }
                return Right(false);
            }
            StaticStruct(*) | StaticEnum(*) => {
                cx.span_bug(span, "CmpOptions used for a static method.");
            }
        };

        match options.rearrange_struct_fields(cx, fields_list) {
            Some(substr) => Left(substr),
            None => Right(false)
        }
    }
}

impl CmpOptions_ {
    // extract the fields from `<option>(<fields>)` and place them
    // into the appropriate array.  e.g. `test_order(foo, bar)` goes
    // into self.test_order.
    fn parse_inner(&mut self, cx: @ExtCtxt,
                   trait_name: &str, allow_ignore: bool, allow_reverse: bool,
                   option_span: span, option_name: &str,
                   fields: &[@ast::MetaItem]) -> bool {
        // FIXME #7930
        let (test_order, ignore, reverse) = match *self {
            CmpOptions_ {
                test_order: ref mut test_order,
                ignore: ref mut ignore,
                reverse: ref mut reverse,
                _
            } => (test_order, ignore, reverse)
        };

        // work out the vec to which the current option corresponds
        // too, if the current option is allowed by the caller, and
        // which vecs any options occuring in this one cannot occur in
        // (e.g. cannot ignore and reverse a field at the same time).
        let (append_to, allowed, disjoint_with) = match option_name {
            "test_order" => (test_order, true,  // always allowed
                             &[("ignore", ignore.as_slice())]),
            "ignore"     => (ignore, allow_ignore,
                             &[("test_order", test_order.as_slice()),
                               ("reverse", reverse.as_slice())]),
            "reverse"    => (reverse, allow_reverse,
                             &[("ignore", ignore.as_slice())]),
            _            => {
                cx.span_err(option_span,
                            fmt!("unrecognised option name: `%s`", option_name));
                cx.span_note(option_span,
                             fmt!("`#[deriving(%s)]` only accepts %s", trait_name,
                                  format_allowed(allow_ignore, allow_reverse, "")));
                return false;
            }
        };

        if !allowed {
            cx.span_err(option_span,
                        fmt!("illegal option: `#[deriving(%s)]` does not allow `%s`",
                             trait_name, option_name));
            return false;
        }

        // Trait(option(<field>, ...))
        let mut no_errors = true;
        for field in fields.iter() {
            match field.node {
                ast::MetaWord(field_name) => {
                    let id = cx.ident_of(field_name);

                    // check for duplicates
                    match append_to.iter().find_(|& &(other_id, _)| id == other_id) {
                        Some(&(_, span)) => { // it's a duplicate!
                            cx.span_err(field.span,
                                        fmt!("field `%s` occurs more than once", field_name));
                            cx.span_note(span, "previous occurrence");
                            no_errors = false;
                            loop;
                        }
                        None => {}
                    }

                    // check that it hasn't already occurred in an
                    // option it shouldn't've, i.e. can't `ignore` and
                    // `reverse` a field at the same time
                    for &(other_name, ref test_against) in disjoint_with.iter() {
                        match test_against.iter().find_(|& &(other_id, _)| id == other_id) {
                            Some(&(_, span)) => {
                                cx.span_err(field.span,
                                            fmt!("field `%s` cannot occur in both `%s` and `%s`",
                                                 field_name, option_name, other_name));
                                cx.span_note(span, "previous occurrence");
                                no_errors = false;
                                loop;
                            }
                            None => {}
                        }
                    }

                    // all ok.
                    append_to.push((id, field.span))
                }
                _ => {
                    cx.span_err(
                        field.span,
                        fmt!("invalid value: `%s` only accepts a list of field names",
                             option_name));
                    no_errors = false;
                }
            }
        }

        no_errors
    }

    /// Only calls `display` if this hasn't been called previously,
    /// i.e. it restricts errors to be shown once, even if
    /// rearrange_struct_fields is called multiple times.
    fn oneshot_error(&mut self, displayer: &fn()) {
        if !self.shown_errors {
            displayer();
            self.shown_errors = true;
        }
    }

    /// Rearrange the vec of fields of a struct with named
    /// fields. Calling this with None for one of the idents will
    /// crash the compiler.
    fn rearrange_struct_fields(&mut self,
                               cx: @ExtCtxt,
                               fields_list: &[(Option<ident>, @expr, ~[@expr])])
        -> Option<SubstructureFields<'static>> {
        // collect the errors rather than showing them incrementally,
        // because we need to print them in one go inside
        // `show_errors_once`.
        let mut errors = ~[];

        // check that all the fields referenced actually exist in the
        // struct.
        { // restrict the lifetime of these immutable borrows.
            let mut check_valid = self.test_order.iter()
                .chain_(self.ignore.iter())
                .chain_(self.reverse.iter());

            // use advance rather than shortcircuiting `all` because we
            // can check all the fields at once this way.
            for &(id, span) in check_valid {
                // run through all the fields of the struct.
                let exists = do fields_list.iter().any |&(some_id, _, _)| {
                    id == some_id.get()
                };
                if !exists {
                    errors.push((span, fmt!("field `%s` does not exist", cx.str_of(id))));
                }
            };
        }
        if !errors.is_empty() {
            do self.oneshot_error {
                for &(span, ref msg) in errors.iter() {
                    cx.span_err(span, *msg)
                }
            }

            return None;
        }

        let mut prioritised_fields = vec::from_elem(self.test_order.len(), None);
        let mut normal_fields = ~[];

        for field in fields_list.iter() {
            match *field {
                (Some(field_id), self_field, [other_field]) => {
                    if self.ignore.iter().any(|&(id, _)| field_id == id) {
                        // it s' ignored, so skip it.
                        loop;
                    }

                    // swap other and self if the field is a reversed
                    // field.
                    let to_insert = if self.reverse.iter().any(|&(id, _)| field_id == id) {
                        (Some(field_id), other_field, ~[self_field])
                    } else {
                        (Some(field_id), self_field, ~[other_field])
                    };

                    match self.test_order.iter().position(|&(id, _)| field_id == id) {
                        Some(i) => prioritised_fields[i] = Some(to_insert),
                        None    => normal_fields.push(to_insert)
                    }
                }
                _ => fail!("Impossible! Somehow a bad field slipped through.")
            }
        }

        let mut out_fields = vec::with_capacity(prioritised_fields.len() + normal_fields.len());
        for field in prioritised_fields.consume_iter() {
            match field {
                Some(x) => out_fields.push(x),
                None => fail!("Impossible! Somehow missing a `test_order` field.")
            }
        }
        out_fields.push_all_move(normal_fields);
        Some(Struct(out_fields))
    }
}

// format the string that says "`test_order()` and `ignore()`" etc,
// based on what is allowed for the current #[deriving()] trait.
fn format_allowed(allow_ignore: bool, allow_reverse: bool, suffix: &str) -> ~str {
    macro_rules! mk(($name:expr) => {
        fmt!("`%s%s`", $name, suffix)
    });
    let mut s = mk!("test_order");
    match (allow_ignore, allow_reverse) {
        (false, false) => {}
        (true,  true) => {
            s.push_str(", ");
            s.push_str(mk!("reverse"));
            s.push_str(" and ");
            s.push_str(mk!("ignore"));
        }
        (true,  false) => {
            s.push_str(" and ");
            s.push_str(mk!("ignore"));
        }
        (false, true) => {
            s.push_str(" and ");
            s.push_str(mk!("reverse"));
        }
    }
    s
}