// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Functions dealing with attributes and meta_items

use core::prelude::*;

use ast;
use ast_util::{spanned, dummy_spanned};
use attr;
use codemap::BytePos;
use diagnostic::span_handler;
use parse::comments::{doc_comment_style, strip_doc_comment_decoration};

use core::cmp;
use core::either::Either;
use core::either;
use core::option;
use core::vec;
use std::map::HashMap;
use std::map;
use std;

/* Constructors */

pub fn mk_name_value_item_str(name: ~str, value: ~str)
                           -> @ast::meta_item {
    let value_lit = dummy_spanned(ast::lit_str(@value));
    return mk_name_value_item(name, value_lit);
}

pub fn mk_name_value_item(name: ~str, +value: ast::lit)
        -> @ast::meta_item {
    return @dummy_spanned(ast::meta_name_value(name, value));
}

pub fn mk_list_item(name: ~str, +items: ~[@ast::meta_item]) ->
   @ast::meta_item {
    return @dummy_spanned(ast::meta_list(name, items));
}

pub fn mk_word_item(name: ~str) -> @ast::meta_item {
    return @dummy_spanned(ast::meta_word(name));
}

pub fn mk_attr(item: @ast::meta_item) -> ast::attribute {
    dummy_spanned(ast::attribute_ { style: ast::attr_inner,
                                    value: *item,
                                    is_sugared_doc: false })
}

pub fn mk_sugared_doc_attr(text: ~str,
                           +lo: BytePos, +hi: BytePos) -> ast::attribute {
    let lit = spanned(lo, hi, ast::lit_str(@text));
    let attr = ast::attribute_ {
        style: doc_comment_style(text),
        value: spanned(lo, hi, ast::meta_name_value(~"doc", lit)),
        is_sugared_doc: true
    };
    spanned(lo, hi, attr)
}

/* Conversion */

pub fn attr_meta(attr: ast::attribute) -> @ast::meta_item {
    @attr.node.value
}

// Get the meta_items from inside a vector of attributes
pub fn attr_metas(attrs: ~[ast::attribute]) -> ~[@ast::meta_item] {
    do attrs.map |a| { attr_meta(*a) }
}

pub fn desugar_doc_attr(attr: &ast::attribute) -> ast::attribute {
    if attr.node.is_sugared_doc {
        let comment = get_meta_item_value_str(@attr.node.value).get();
        let meta = mk_name_value_item_str(~"doc",
                                     strip_doc_comment_decoration(comment));
        return mk_attr(meta);
    } else {
        *attr
    }
}

/* Accessors */

pub fn get_attr_name(attr: ast::attribute) -> ~str {
    get_meta_item_name(@attr.node.value)
}

pub fn get_meta_item_name(meta: @ast::meta_item) -> ~str {
    match meta.node {
      ast::meta_word(ref n) => (*n),
      ast::meta_name_value(ref n, _) => (*n),
      ast::meta_list(ref n, _) => (*n)
    }
}

/**
 * Gets the string value if the meta_item is a meta_name_value variant
 * containing a string, otherwise none
 */
pub fn get_meta_item_value_str(meta: @ast::meta_item) -> Option<~str> {
    match meta.node {
        ast::meta_name_value(_, v) => match v.node {
            ast::lit_str(s) => option::Some(*s),
            _ => option::None
        },
        _ => option::None
    }
}

/// Gets a list of inner meta items from a list meta_item type
pub fn get_meta_item_list(meta: @ast::meta_item)
                       -> Option<~[@ast::meta_item]> {
    match meta.node {
      ast::meta_list(_, l) => option::Some(/* FIXME (#2543) */ copy l),
      _ => option::None
    }
}

/**
 * If the meta item is a nam-value type with a string value then returns
 * a tuple containing the name and string value, otherwise `none`
 */
pub fn get_name_value_str_pair(item: @ast::meta_item)
                            -> Option<(~str, ~str)> {
    match attr::get_meta_item_value_str(item) {
      Some(ref value) => {
        let name = attr::get_meta_item_name(item);
        Some((name, (*value)))
      }
      None => None
    }
}


/* Searching */

/// Search a list of attributes and return only those with a specific name
pub fn find_attrs_by_name(attrs: &[ast::attribute], name: &str) ->
   ~[ast::attribute] {
    let filter: &fn(a: &ast::attribute) -> Option<ast::attribute> = |a| {
        if name == get_attr_name(*a) {
            option::Some(*a)
        } else {
            option::None
        }
    };
    return vec::filter_map(attrs, filter);
}

/// Search a list of meta items and return only those with a specific name
pub fn find_meta_items_by_name(metas: &[@ast::meta_item], name: &str) ->
   ~[@ast::meta_item] {
    let mut rs = ~[];
    for metas.each |mi| {
        if name == get_meta_item_name(*mi) {
            rs.push(*mi)
        }
    }
    rs
}

/**
 * Returns true if a list of meta items contains another meta item. The
 * comparison is performed structurally.
 */
pub fn contains(haystack: ~[@ast::meta_item], needle: @ast::meta_item)
             -> bool {
    for haystack.each |item| {
        if eq(*item, needle) { return true; }
    }
    return false;
}

fn eq(a: @ast::meta_item, b: @ast::meta_item) -> bool {
    return match a.node {
          ast::meta_word(ref na) => match b.node {
            ast::meta_word(ref nb) => (*na) == (*nb),
            _ => false
          },
          ast::meta_name_value(ref na, va) => match b.node {
            ast::meta_name_value(ref nb, vb) => {
                (*na) == (*nb) && va.node == vb.node
            }
            _ => false
          },
          ast::meta_list(*) => {

            // ~[Fixme-sorting]
            // FIXME (#607): Needs implementing
            // This involves probably sorting the list by name and
            // meta_item variant
            fail ~"unimplemented meta_item variant"
          }
        }
}

pub fn contains_name(metas: &[@ast::meta_item], name: &str) -> bool {
    let matches = find_meta_items_by_name(metas, name);
    return vec::len(matches) > 0u;
}

pub fn attrs_contains_name(attrs: &[ast::attribute], name: &str) -> bool {
    !find_attrs_by_name(attrs, name).is_empty()
}

pub fn first_attr_value_str_by_name(attrs: ~[ast::attribute], name: ~str)
                                 -> Option<~str> {

    let mattrs = find_attrs_by_name(attrs, name);
    if vec::len(mattrs) > 0u {
        return get_meta_item_value_str(attr_meta(mattrs[0]));
    }
    return option::None;
}

fn last_meta_item_by_name(items: ~[@ast::meta_item], name: ~str)
    -> Option<@ast::meta_item> {

    let items = attr::find_meta_items_by_name(items, name);
    vec::last_opt(items)
}

pub fn last_meta_item_value_str_by_name(items: ~[@ast::meta_item], name: ~str)
                                     -> Option<~str> {

    match last_meta_item_by_name(items, name) {
      Some(item) => match attr::get_meta_item_value_str(item) {
        Some(ref value) => Some((*value)),
        None => None
      },
      None => None
    }
}

pub fn last_meta_item_list_by_name(items: ~[@ast::meta_item], name: ~str)
    -> Option<~[@ast::meta_item]> {

    match last_meta_item_by_name(items, name) {
      Some(item) => attr::get_meta_item_list(item),
      None => None
    }
}


/* Higher-level applications */

// FIXME (#607): This needs to sort by meta_item variant in addition to
// the item name (See [Fixme-sorting])
pub fn sort_meta_items(+items: ~[@ast::meta_item]) -> ~[@ast::meta_item] {
    pure fn lteq(ma: &@ast::meta_item, mb: &@ast::meta_item) -> bool {
        pure fn key(m: &ast::meta_item) -> ~str {
            match m.node {
              ast::meta_word(ref name) => (*name),
              ast::meta_name_value(ref name, _) => (*name),
              ast::meta_list(ref name, _) => (*name)
            }
        }
        key(*ma) <= key(*mb)
    }

    // This is sort of stupid here, converting to a vec of mutables and back
    let mut v: ~[@ast::meta_item] = vec::cast_to_mut(items);
    std::sort::quick_sort(v, lteq);
    vec::cast_from_mut(move v)
}

pub fn remove_meta_items_by_name(items: ~[@ast::meta_item], name: ~str) ->
   ~[@ast::meta_item] {

    return vec::filter_map(items, |item| {
        if get_meta_item_name(*item) != name {
            option::Some(/* FIXME (#2543) */ copy *item)
        } else {
            option::None
        }
    });
}

/**
 * From a list of crate attributes get only the meta_items that affect crate
 * linkage
 */
pub fn find_linkage_metas(attrs: &[ast::attribute]) -> ~[@ast::meta_item] {
    do find_attrs_by_name(attrs, ~"link").flat_map |attr| {
        match attr.node.value.node {
            ast::meta_list(_, items) => /* FIXME (#2543) */ copy items,
            _ => ~[]
        }
    }
}

pub fn foreign_abi(attrs: ~[ast::attribute])
                -> Either<~str, ast::foreign_abi> {
    return match attr::first_attr_value_str_by_name(attrs, ~"abi") {
      option::None => {
        either::Right(ast::foreign_abi_cdecl)
      }
      option::Some(~"rust-intrinsic") => {
        either::Right(ast::foreign_abi_rust_intrinsic)
      }
      option::Some(~"cdecl") => {
        either::Right(ast::foreign_abi_cdecl)
      }
      option::Some(~"stdcall") => {
        either::Right(ast::foreign_abi_stdcall)
      }
      option::Some(ref t) => {
        either::Left(~"unsupported abi: " + (*t))
      }
    };
}

pub enum inline_attr {
    ia_none,
    ia_hint,
    ia_always,
    ia_never,
}

impl inline_attr : cmp::Eq {
    pure fn eq(&self, other: &inline_attr) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &inline_attr) -> bool { !(*self).eq(other) }
}

/// True if something like #[inline] is found in the list of attrs.
pub fn find_inline_attr(attrs: &[ast::attribute]) -> inline_attr {
    // FIXME (#2809)---validate the usage of #[inline] and #[inline(always)]
    do vec::foldl(ia_none, attrs) |ia,attr| {
        match attr.node.value.node {
          ast::meta_word(~"inline") => ia_hint,
          ast::meta_list(~"inline", items) => {
            if !vec::is_empty(find_meta_items_by_name(items, ~"always")) {
                ia_always
            } else if !vec::is_empty(
                find_meta_items_by_name(items, ~"never")) {
                ia_never
            } else {
                ia_hint
            }
          }
          _ => ia
        }
    }
}


pub fn require_unique_names(diagnostic: span_handler,
                            metas: &[@ast::meta_item]) {
    let map = map::HashMap();
    for metas.each |meta| {
        let name = get_meta_item_name(*meta);

        // FIXME: How do I silence the warnings? --pcw (#2619)
        if map.contains_key(name) {
            diagnostic.span_fatal(meta.span,
                                  fmt!("duplicate meta item `%s`", name));
        }
        map.insert(name, ());
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
