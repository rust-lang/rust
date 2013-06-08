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
use codemap::{spanned, dummy_spanned};
use attr;
use codemap::BytePos;
use diagnostic::span_handler;
use parse::comments::{doc_comment_style, strip_doc_comment_decoration};

use core::iterator::IteratorUtil;
use core::hashmap::HashSet;
use core::vec;
use extra;

/* Constructors */

pub fn mk_name_value_item_str(name: @~str, value: @~str)
                           -> @ast::meta_item {
    let value_lit = dummy_spanned(ast::lit_str(value));
    mk_name_value_item(name, value_lit)
}

pub fn mk_name_value_item(name: @~str, value: ast::lit)
        -> @ast::meta_item {
    @dummy_spanned(ast::meta_name_value(name, value))
}

pub fn mk_list_item(name: @~str, items: ~[@ast::meta_item]) ->
   @ast::meta_item {
    @dummy_spanned(ast::meta_list(name, items))
}

pub fn mk_word_item(name: @~str) -> @ast::meta_item {
    @dummy_spanned(ast::meta_word(name))
}

pub fn mk_attr(item: @ast::meta_item) -> ast::attribute {
    dummy_spanned(ast::attribute_ { style: ast::attr_inner,
                                    value: item,
                                    is_sugared_doc: false })
}

pub fn mk_sugared_doc_attr(text: ~str,
                           lo: BytePos, hi: BytePos) -> ast::attribute {
    let style = doc_comment_style(text);
    let lit = spanned(lo, hi, ast::lit_str(@text));
    let attr = ast::attribute_ {
        style: style,
        value: @spanned(lo, hi, ast::meta_name_value(@~"doc", lit)),
        is_sugared_doc: true
    };
    spanned(lo, hi, attr)
}

/* Conversion */

pub fn attr_meta(attr: ast::attribute) -> @ast::meta_item {
    attr.node.value
}

// Get the meta_items from inside a vector of attributes
pub fn attr_metas(attrs: &[ast::attribute]) -> ~[@ast::meta_item] {
    do attrs.map |a| { attr_meta(*a) }
}

pub fn desugar_doc_attr(attr: &ast::attribute) -> ast::attribute {
    if attr.node.is_sugared_doc {
        let comment = get_meta_item_value_str(attr.node.value).get();
        let meta = mk_name_value_item_str(@~"doc",
                                     @strip_doc_comment_decoration(*comment));
        mk_attr(meta)
    } else {
        *attr
    }
}

/* Accessors */

pub fn get_attr_name(attr: &ast::attribute) -> @~str {
    get_meta_item_name(attr.node.value)
}

pub fn get_meta_item_name(meta: @ast::meta_item) -> @~str {
    match meta.node {
        ast::meta_word(n) => n,
        ast::meta_name_value(n, _) => n,
        ast::meta_list(n, _) => n,
    }
}

/**
 * Gets the string value if the meta_item is a meta_name_value variant
 * containing a string, otherwise none
 */
pub fn get_meta_item_value_str(meta: @ast::meta_item) -> Option<@~str> {
    match meta.node {
        ast::meta_name_value(_, v) => {
            match v.node {
                ast::lit_str(s) => Some(s),
                _ => None,
            }
        },
        _ => None
    }
}

/// Gets a list of inner meta items from a list meta_item type
pub fn get_meta_item_list(meta: @ast::meta_item)
                       -> Option<~[@ast::meta_item]> {
    match meta.node {
        ast::meta_list(_, ref l) => Some(/* FIXME (#2543) */ copy *l),
        _ => None
    }
}

/**
 * If the meta item is a nam-value type with a string value then returns
 * a tuple containing the name and string value, otherwise `none`
 */
pub fn get_name_value_str_pair(item: @ast::meta_item)
                            -> Option<(@~str, @~str)> {
    match attr::get_meta_item_value_str(item) {
      Some(value) => {
        let name = attr::get_meta_item_name(item);
        Some((name, value))
      }
      None => None
    }
}


/* Searching */

/// Search a list of attributes and return only those with a specific name
pub fn find_attrs_by_name(attrs: &[ast::attribute], name: &str) ->
   ~[ast::attribute] {
    do vec::filter_mapped(attrs) |a| {
        if name == *get_attr_name(a) {
            Some(*a)
        } else {
            None
        }
    }
}

/// Search a list of meta items and return only those with a specific name
pub fn find_meta_items_by_name(metas: &[@ast::meta_item], name: &str) ->
   ~[@ast::meta_item] {
    let mut rs = ~[];
    for metas.each |mi| {
        if name == *get_meta_item_name(*mi) {
            rs.push(*mi)
        }
    }
    rs
}

/**
 * Returns true if a list of meta items contains another meta item. The
 * comparison is performed structurally.
 */
pub fn contains(haystack: &[@ast::meta_item],
                needle: @ast::meta_item) -> bool {
    for haystack.each |item| {
        if eq(*item, needle) { return true; }
    }
    return false;
}

fn eq(a: @ast::meta_item, b: @ast::meta_item) -> bool {
    match a.node {
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
        ast::meta_list(ref na, ref misa) => match b.node {
            ast::meta_list(ref nb, ref misb) => {
                if na != nb { return false; }
                for misa.each |mi| {
                    if !misb.contains(mi) { return false; }
                }
                true
            }
            _ => false
        }
    }
}

pub fn contains_name(metas: &[@ast::meta_item], name: &str) -> bool {
    let matches = find_meta_items_by_name(metas, name);
    matches.len() > 0u
}

pub fn attrs_contains_name(attrs: &[ast::attribute], name: &str) -> bool {
    !find_attrs_by_name(attrs, name).is_empty()
}

pub fn first_attr_value_str_by_name(attrs: &[ast::attribute], name: &str)
                                 -> Option<@~str> {

    let mattrs = find_attrs_by_name(attrs, name);
    if mattrs.len() > 0 {
        get_meta_item_value_str(attr_meta(mattrs[0]))
    } else {
        None
    }
}

fn last_meta_item_by_name(items: &[@ast::meta_item], name: &str)
    -> Option<@ast::meta_item> {

    let items = attr::find_meta_items_by_name(items, name);
    items.last_opt().map(|item| **item)
}

pub fn last_meta_item_value_str_by_name(items: &[@ast::meta_item], name: &str)
                                     -> Option<@~str> {

    match last_meta_item_by_name(items, name) {
        Some(item) => {
            match attr::get_meta_item_value_str(item) {
                Some(value) => Some(value),
                None => None
            }
        },
        None => None
    }
}

pub fn last_meta_item_list_by_name(items: ~[@ast::meta_item], name: &str)
    -> Option<~[@ast::meta_item]> {

    match last_meta_item_by_name(items, name) {
      Some(item) => attr::get_meta_item_list(item),
      None => None
    }
}


/* Higher-level applications */

pub fn sort_meta_items(items: &[@ast::meta_item]) -> ~[@ast::meta_item] {
    // This is sort of stupid here, converting to a vec of mutables and back
    let mut v = vec::to_owned(items);
    do extra::sort::quick_sort(v) |ma, mb| {
        get_meta_item_name(*ma) <= get_meta_item_name(*mb)
    }

    // There doesn't seem to be a more optimal way to do this
    do v.map |m| {
        match m.node {
            ast::meta_list(n, ref mis) => {
                @spanned {
                    node: ast::meta_list(n, sort_meta_items(*mis)),
                    .. /*bad*/ copy **m
                }
            }
            _ => /*bad*/ copy *m
        }
    }
}

pub fn remove_meta_items_by_name(items: ~[@ast::meta_item], name: &str) ->
   ~[@ast::meta_item] {

    return vec::filter_mapped(items, |item| {
        if name != *get_meta_item_name(*item) {
            Some(*item)
        } else {
            None
        }
    });
}

/**
 * From a list of crate attributes get only the meta_items that affect crate
 * linkage
 */
pub fn find_linkage_metas(attrs: &[ast::attribute]) -> ~[@ast::meta_item] {
    do find_attrs_by_name(attrs, "link").flat_map |attr| {
        match attr.node.value.node {
            ast::meta_list(_, ref items) => /* FIXME (#2543) */ copy *items,
            _ => ~[]
        }
    }
}

#[deriving(Eq)]
pub enum inline_attr {
    ia_none,
    ia_hint,
    ia_always,
    ia_never,
}

/// True if something like #[inline] is found in the list of attrs.
pub fn find_inline_attr(attrs: &[ast::attribute]) -> inline_attr {
    // FIXME (#2809)---validate the usage of #[inline] and #[inline(always)]
    do attrs.iter().fold(ia_none) |ia,attr| {
        match attr.node.value.node {
          ast::meta_word(@~"inline") => ia_hint,
          ast::meta_list(@~"inline", ref items) => {
            if !find_meta_items_by_name(*items, "always").is_empty() {
                ia_always
            } else if !find_meta_items_by_name(*items, "never").is_empty() {
                ia_never
            } else {
                ia_hint
            }
          }
          _ => ia
        }
    }
}


pub fn require_unique_names(diagnostic: @span_handler,
                            metas: &[@ast::meta_item]) {
    let mut set = HashSet::new();
    for metas.each |meta| {
        let name = get_meta_item_name(*meta);

        // FIXME: How do I silence the warnings? --pcw (#2619)
        if !set.insert(name) {
            diagnostic.span_fatal(meta.span,
                                  fmt!("duplicate meta item `%s`", *name));
        }
    }
}
