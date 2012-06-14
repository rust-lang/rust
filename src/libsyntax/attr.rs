// Functions dealing with attributes and meta_items

import std::map;
import std::map::hashmap;
import either::either;
import diagnostic::span_handler;
import ast_util::dummy_spanned;

// Constructors
export mk_name_value_item_str;
export mk_name_value_item;
export mk_list_item;
export mk_word_item;
export mk_attr;

// Conversion
export attr_meta;
export attr_metas;

// Accessors
export get_attr_name;
export get_meta_item_name;
export get_meta_item_value_str;
export get_meta_item_list;
export get_name_value_str_pair;

// Searching
export find_attrs_by_name;
export find_meta_items_by_name;
export contains;
export contains_name;
export attrs_contains_name;
export first_attr_value_str_by_name;
export last_meta_item_value_str_by_name;
export last_meta_item_list_by_name;

// Higher-level applications
export sort_meta_items;
export remove_meta_items_by_name;
export find_linkage_attrs;
export find_linkage_metas;
export native_abi;
export inline_attr;
export find_inline_attr;
export require_unique_names;

/* Constructors */

fn mk_name_value_item_str(+name: ast::ident, +value: str) -> @ast::meta_item {
    let value_lit = dummy_spanned(ast::lit_str(@value));
    ret mk_name_value_item(name, value_lit);
}

fn mk_name_value_item(+name: ast::ident, +value: ast::lit)
        -> @ast::meta_item {
    ret @dummy_spanned(ast::meta_name_value(name, value));
}

fn mk_list_item(+name: ast::ident, +items: [@ast::meta_item]) ->
   @ast::meta_item {
    ret @dummy_spanned(ast::meta_list(name, items));
}

fn mk_word_item(+name: ast::ident) -> @ast::meta_item {
    ret @dummy_spanned(ast::meta_word(name));
}

fn mk_attr(item: @ast::meta_item) -> ast::attribute {
    ret dummy_spanned({style: ast::attr_inner, value: *item});
}


/* Conversion */

fn attr_meta(attr: ast::attribute) -> @ast::meta_item { @attr.node.value }

// Get the meta_items from inside a vector of attributes
fn attr_metas(attrs: [ast::attribute]) -> [@ast::meta_item] {
    let mut mitems = [];
    for attrs.each {|a| mitems += [attr_meta(a)]; }
    ret mitems;
}


/* Accessors */

fn get_attr_name(attr: ast::attribute) -> ast::ident {
    get_meta_item_name(@attr.node.value)
}

fn get_meta_item_name(meta: @ast::meta_item) -> ast::ident {
    alt meta.node {
      ast::meta_word(n) { /* FIXME bad */ copy n }
      ast::meta_name_value(n, _) { /* FIXME bad */ copy n }
      ast::meta_list(n, _) { /* FIXME bad */ copy n }
    }
}

#[doc = "
Gets the string value if the meta_item is a meta_name_value variant
containing a string, otherwise none
"]
fn get_meta_item_value_str(meta: @ast::meta_item) -> option<@str> {
    alt meta.node {
      ast::meta_name_value(_, v) {
        alt v.node {
            ast::lit_str(s) {
                option::some(s)
            }
            _ {
                option::none
            }
        }
      }
      _ { option::none }
    }
}

#[doc = "Gets a list of inner meta items from a list meta_item type"]
fn get_meta_item_list(meta: @ast::meta_item) -> option<[@ast::meta_item]> {
    alt meta.node {
      ast::meta_list(_, l) { option::some(/* FIXME bad */ copy l) }
      _ { option::none }
    }
}

#[doc = "
If the meta item is a nam-value type with a string value then returns
a tuple containing the name and string value, otherwise `none`
"]
fn get_name_value_str_pair(
    item: @ast::meta_item
) -> option<(ast::ident, @str)> {
    alt attr::get_meta_item_value_str(item) {
      some(value) {
        let name = attr::get_meta_item_name(item);
        some((name, value))
      }
      none { none }
    }
}


/* Searching */

#[doc = "
Search a list of attributes and return only those with a specific name
"]
fn find_attrs_by_name(attrs: [ast::attribute], +name: str) ->
   [ast::attribute] {
    let filter = (
        fn@(a: ast::attribute) -> option<ast::attribute> {
            if *get_attr_name(a) == name {
                option::some(a)
            } else { option::none }
        }
    );
    ret vec::filter_map(attrs, filter);
}

#[doc = "
Searcha list of meta items and return only those with a specific name
"]
fn find_meta_items_by_name(metas: [@ast::meta_item], +name: str) ->
   [@ast::meta_item] {
    let filter = fn@(&&m: @ast::meta_item) -> option<@ast::meta_item> {
        if *get_meta_item_name(m) == name {
            option::some(m)
        } else { option::none }
    };
    ret vec::filter_map(metas, filter);
}

#[doc = "
Returns true if a list of meta items contains another meta item. The
comparison is performed structurally.
"]
fn contains(haystack: [@ast::meta_item], needle: @ast::meta_item) -> bool {
    #debug("looking for %s",
           print::pprust::meta_item_to_str(*needle));
    for haystack.each {|item|
        #debug("looking in %s",
               print::pprust::meta_item_to_str(*item));
        if eq(item, needle) { #debug("found it!"); ret true; }
    }
    #debug("found it not :(");
    ret false;
}

fn eq(a: @ast::meta_item, b: @ast::meta_item) -> bool {
    ret alt a.node {
          ast::meta_word(na) {
            alt b.node { ast::meta_word(nb) { na == nb } _ { false } }
          }
          ast::meta_name_value(na, va) {
            alt b.node {
              ast::meta_name_value(nb, vb) { na == nb && va.node == vb.node }
              _ { false }
            }
          }
          ast::meta_list(na, la) {

            // [Fixme-sorting]
            // FIXME (#607): Needs implementing
            // This involves probably sorting the list by name and
            // meta_item variant
            fail "unimplemented meta_item variant"
          }
        }
}

fn contains_name(metas: [@ast::meta_item], +name: str) -> bool {
    let matches = find_meta_items_by_name(metas, name);
    ret vec::len(matches) > 0u;
}

fn attrs_contains_name(attrs: [ast::attribute], +name: str) -> bool {
    vec::is_not_empty(find_attrs_by_name(attrs, name))
}

fn first_attr_value_str_by_name(attrs: [ast::attribute], +name: str)
    -> option<@str> {
    let mattrs = find_attrs_by_name(attrs, name);
    if vec::len(mattrs) > 0u {
        ret get_meta_item_value_str(attr_meta(mattrs[0]));
    }
    ret option::none;
}

fn last_meta_item_by_name(
    items: [@ast::meta_item],
    +name: str
) -> option<@ast::meta_item> {
    let items = attr::find_meta_items_by_name(items, name);
    vec::last_opt(items)
}

fn last_meta_item_value_str_by_name(
    items: [@ast::meta_item],
    +name: str
) -> option<@str> {
    alt last_meta_item_by_name(items, name) {
      some(item) {
        alt attr::get_meta_item_value_str(item) {
          some(value) { some(value) }
          none { none }
        }
      }
      none { none }
    }
}

fn last_meta_item_list_by_name(
    items: [@ast::meta_item],
    +name: str
) -> option<[@ast::meta_item]> {
    alt last_meta_item_by_name(items, name) {
      some(item) {
        attr::get_meta_item_list(item)
      }
      none { none }
    }
}


/* Higher-level applications */

// FIXME: This needs to sort by meta_item variant in addition to the item name
// (See [Fixme-sorting])
fn sort_meta_items(+items: [@ast::meta_item]) -> [@ast::meta_item] {
    fn lteq(&&ma: @ast::meta_item, &&mb: @ast::meta_item) -> bool {
        fn key(m: @ast::meta_item) -> ast::ident {
            alt m.node {
              ast::meta_word(name) { /* FIXME bad */ copy name }
              ast::meta_name_value(name, _) { /* FIXME bad */ copy name }
              ast::meta_list(name, _) { /* FIXME bad */ copy name }
            }
        }
        ret key(ma) <= key(mb);
    }

    // This is sort of stupid here, converting to a vec of mutables and back
    let v: [mut @ast::meta_item] = vec::to_mut(items);
    std::sort::quick_sort(lteq, v);
    ret vec::from_mut(v);
}

fn remove_meta_items_by_name(items: [@ast::meta_item], name: ast::ident) ->
   [@ast::meta_item] {

    ret vec::filter_map(items, {
        |item|
        if get_meta_item_name(item) != name {
            option::some(/* FIXME bad */ copy item)
        } else {
            option::none
        }
    });
}

fn find_linkage_attrs(attrs: [ast::attribute]) -> [ast::attribute] {
    let mut found = [];
    for find_attrs_by_name(attrs, "link").each {|attr|
        alt attr.node.value.node {
          ast::meta_list(_, _) { found += [attr] }
          _ { #debug("ignoring link attribute that has incorrect type"); }
        }
    }
    ret found;
}

#[doc = "
From a list of crate attributes get only the meta_items that impact crate
linkage
"]
fn find_linkage_metas(attrs: [ast::attribute]) -> [@ast::meta_item] {
    find_linkage_attrs(attrs).flat_map {|attr|
        alt check attr.node.value.node {
          ast::meta_list(_, items) { /* FIXME bad */ copy items }
        }
    }
}

fn native_abi(attrs: [ast::attribute]) -> either<str, ast::native_abi> {
    ret alt attr::first_attr_value_str_by_name(attrs, "abi") {
      option::none {
        either::right(ast::native_abi_cdecl)
      }
      option::some(@"rust-intrinsic") {
        either::right(ast::native_abi_rust_intrinsic)
      }
      option::some(@"cdecl") {
        either::right(ast::native_abi_cdecl)
      }
      option::some(@"stdcall") {
        either::right(ast::native_abi_stdcall)
      }
      option::some(t) {
        either::left("unsupported abi: " + *t)
      }
    };
}

enum inline_attr {
    ia_none,
    ia_hint,
    ia_always
}

#[doc = "True if something like #[inline] is found in the list of attrs."]
fn find_inline_attr(attrs: [ast::attribute]) -> inline_attr {
    // TODO---validate the usage of #[inline] and #[inline(always)]
    vec::foldl(ia_none, attrs) {|ia,attr|
        alt attr.node.value.node {
          ast::meta_word(@"inline") { ia_hint }
          ast::meta_list(@"inline", items) {
            if !vec::is_empty(find_meta_items_by_name(items, "always")) {
                ia_always
            } else {
                ia_hint
            }
          }
          _ { ia }
        }
    }
}


fn require_unique_names(diagnostic: span_handler,
                        metas: [@ast::meta_item]) {
    let map = map::str_hash();
    for metas.each {|meta|
        let name = get_meta_item_name(meta);

        // FIXME: How do I silence the warnings? --pcw
        if map.contains_key(*name) {
            diagnostic.span_fatal(meta.span,
                                  #fmt["duplicate meta item `%s`", *name]);
        }
        map.insert(*name, ());
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
