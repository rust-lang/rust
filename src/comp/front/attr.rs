// Functions dealing with attributes and meta_items

import std::vec;
import std::str;
import std::map;
import std::option;
import syntax::ast;
import util::common;
import driver::session;

export attr_meta;
export attr_metas;
export find_linkage_metas;
export find_attrs_by_name;
export find_meta_items_by_name;
export contains;
export contains_name;
export sort_meta_items;
export remove_meta_items_by_name;
export require_unique_names;
export get_attr_name;
export get_meta_item_name;
export get_meta_item_value_str;
export mk_name_value_item_str;
export mk_name_value_item;
export mk_list_item;
export mk_word_item;
export mk_attr;

// From a list of crate attributes get only the meta_items that impact crate
// linkage
fn find_linkage_metas(attrs: &[ast::attribute]) -> [@ast::meta_item] {
    let metas: [@ast::meta_item] = ~[];
    for attr: ast::attribute in find_attrs_by_name(attrs, "link") {
        alt attr.node.value.node {
          ast::meta_list(_, items) { metas += items; }
          _ { log "ignoring link attribute that has incorrect type"; }
        }
    }
    ret metas;
}

// Search a list of attributes and return only those with a specific name
fn find_attrs_by_name(attrs: &[ast::attribute], name: ast::ident) ->
   [ast::attribute] {
    let filter =
        bind fn (a: &ast::attribute, name: ast::ident) ->
                option::t[ast::attribute] {
                 if get_attr_name(a) == name {
                     option::some(a)
                 } else { option::none }
             }(_, name);
    ret vec::filter_map(filter, attrs);
}

fn get_attr_name(attr: &ast::attribute) -> ast::ident {
    get_meta_item_name(@attr.node.value)
}

fn find_meta_items_by_name(metas: &[@ast::meta_item], name: ast::ident) ->
   [@ast::meta_item] {
    let filter =
        bind fn (m: &@ast::meta_item, name: ast::ident) ->
                option::t[@ast::meta_item] {
                 if get_meta_item_name(m) == name {
                     option::some(m)
                 } else { option::none }
             }(_, name);
    ret vec::filter_map(filter, metas);
}

fn get_meta_item_name(meta: &@ast::meta_item) -> ast::ident {
    alt meta.node {
      ast::meta_word(n) { n }
      ast::meta_name_value(n, _) { n }
      ast::meta_list(n, _) { n }
    }
}

// Gets the string value if the meta_item is a meta_name_value variant
// containing a string, otherwise none
fn get_meta_item_value_str(meta: &@ast::meta_item) -> option::t[str] {
    alt meta.node {
      ast::meta_name_value(_, v) {
        alt v.node {
          ast::lit_str(s, _) { option::some(s) }
          _ { option::none }
        }
      }
      _ { option::none }
    }
}

fn attr_meta(attr: &ast::attribute) -> @ast::meta_item { @attr.node.value }

// Get the meta_items from inside a vector of attributes
fn attr_metas(attrs: &[ast::attribute]) -> [@ast::meta_item] {
    let mitems = ~[];
    for a: ast::attribute in attrs { mitems += ~[attr_meta(a)]; }
    ret mitems;
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

            // FIXME (#607): Needs implementing
            // This involves probably sorting the list by name and
            // meta_item variant
            fail "unimplemented meta_item variant"
          }
        }
}

fn contains(haystack: &[@ast::meta_item], needle: @ast::meta_item) -> bool {
    log #fmt("looking for %s",
             syntax::print::pprust::meta_item_to_str(*needle));
    for item: @ast::meta_item in haystack {
        log #fmt("looking in %s",
                 syntax::print::pprust::meta_item_to_str(*item));
        if eq(item, needle) { log "found it!"; ret true; }
    }
    log "found it not :(";
    ret false;
}

fn contains_name(metas: &[@ast::meta_item], name: ast::ident) -> bool {
    let matches = find_meta_items_by_name(metas, name);
    ret vec::len(matches) > 0u;
}

// FIXME: This needs to sort by meta_item variant in addition to the item name
fn sort_meta_items(items: &[@ast::meta_item]) -> [@ast::meta_item] {
    fn lteq(ma: &@ast::meta_item, mb: &@ast::meta_item) -> bool {
        fn key(m: &@ast::meta_item) -> ast::ident {
            alt m.node {
              ast::meta_word(name) { name }
              ast::meta_name_value(name, _) { name }
              ast::meta_list(name, _) { name }
            }
        }
        ret key(ma) <= key(mb);
    }

    // This is sort of stupid here, converting to a vec of mutables and back
    let v: [mutable @ast::meta_item] = ~[mutable];
    for mi: @ast::meta_item in items { v += ~[mutable mi]; }

    std::sort::quick_sort(lteq, v);

    let v2: [@ast::meta_item] = ~[];
    for mi: @ast::meta_item in v { v2 += ~[mi]; }
    ret v2;
}

fn remove_meta_items_by_name(items: &[@ast::meta_item], name: str) ->
   [@ast::meta_item] {

    let filter =
        bind fn (item: &@ast::meta_item, name: str) ->
                option::t[@ast::meta_item] {
                 if get_meta_item_name(item) != name {
                     option::some(item)
                 } else { option::none }
             }(_, name);

    ret vec::filter_map(filter, items);
}

fn require_unique_names(sess: &session::session,
                        metas: &[@ast::meta_item]) {
    let map = map::mk_hashmap[str, ()](str::hash, str::eq);
    for meta: @ast::meta_item in metas {
        let name = get_meta_item_name(meta);
        if map.contains_key(name) {
            sess.span_fatal(meta.span,
                            #fmt("duplicate meta item `%s`", name));
        }
        map.insert(name, ());
    }
}

fn span[T](item: &T) -> ast::spanned[T] {
    ret {node: item, span: ast::mk_sp(0u, 0u)};
}

fn mk_name_value_item_str(name: ast::ident, value: str) -> @ast::meta_item {
    let value_lit = span(ast::lit_str(value, ast::sk_rc));
    ret mk_name_value_item(name, value_lit);
}

fn mk_name_value_item(name: ast::ident, value: ast::lit) -> @ast::meta_item {
    ret @span(ast::meta_name_value(name, value));
}

fn mk_list_item(name: ast::ident, items: &[@ast::meta_item]) ->
   @ast::meta_item {
    ret @span(ast::meta_list(name, items));
}

fn mk_word_item(name: ast::ident) -> @ast::meta_item {
    ret @span(ast::meta_word(name));
}

fn mk_attr(item: @ast::meta_item) -> ast::attribute {
    ret span({style: ast::attr_inner, value: *item});
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
