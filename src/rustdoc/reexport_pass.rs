#[doc = "Finds docs for reexported items and duplicates them"];

import std::map;
import std::map::hashmap;
import std::list;
import syntax::ast;
import syntax::ast_util;
import rustc::util::common;
import rustc::middle::ast_map;
import syntax::visit;
import syntax::codemap;
import rustc::middle::resolve;

export mk_pass;

fn mk_pass() -> pass {
    {
        name: "reexport",
        f: run
    }
}

type def_set = map::set<ast::def_id>;
type def_map = map::hashmap<ast::def_id, doc::itemtag>;
type path_map = map::hashmap<str, [(str, doc::itemtag)]>;

fn run(srv: astsrv::srv, doc: doc::doc) -> doc::doc {

    // First gather the set of defs that are used as reexports
    let def_set = build_reexport_def_set(srv);

    // Now find the docs that go with those defs
    let def_map = build_reexport_def_map(srv, doc, def_set);

    // Now create a map that tells us where to insert the duplicated
    // docs into the existing doc tree
    let path_map = build_reexport_path_map(srv, def_map);

    // Finally update the doc tree
    merge_reexports(doc, path_map)
}

// Hash maps are not sendable so converting them back and forth
// to association lists. Yuck.
fn to_assoc_list<K:copy, V:copy>(
    map: map::hashmap<K, V>
) -> [(K, V)] {

    let mut vec = [];
    for map.each {|k, v|
        vec += [(k, v)];
    }
    ret vec;
}

fn from_assoc_list<K:copy, V:copy>(
    list: [(K, V)],
    new_hash: fn() -> map::hashmap<K, V>
) -> map::hashmap<K, V> {

    let map = new_hash();
    vec::iter(list) {|elt|
        let (k, v) = elt;
        map.insert(k, v);
    }
    ret map;
}

fn from_def_assoc_list<V:copy>(
    list: [(ast::def_id, V)]
) -> map::hashmap<ast::def_id, V> {
    from_assoc_list(list, bind common::new_def_hash())
}

fn from_str_assoc_list<V:copy>(
    list: [(str, V)]
) -> map::hashmap<str, V> {
    from_assoc_list(list, bind map::str_hash())
}

fn build_reexport_def_set(srv: astsrv::srv) -> def_set {
    let assoc_list = astsrv::exec(srv) {|ctxt|
        let def_set = common::new_def_hash();
        for ctxt.exp_map.each {|_id, defs|
            for defs.each {|def|
                if def.reexp {
                    def_set.insert(def.id, ());
                }
            }
        }
        for find_reexport_impls(ctxt).each {|def|
            def_set.insert(def, ());
        }
        to_assoc_list(def_set)
    };

    from_def_assoc_list(assoc_list)
}

fn find_reexport_impls(ctxt: astsrv::ctxt) -> [ast::def_id] {
    let defs = @mut [];
    for_each_reexported_impl(ctxt) {|_mod_id, i|
        *defs += [i.did]
    }
    ret *defs;
}

fn build_reexport_def_map(
    srv: astsrv::srv,
    doc: doc::doc,
    def_set: def_set
) -> def_map {

    type ctxt = {
        srv: astsrv::srv,
        def_set: def_set,
        def_map: def_map
    };

    let ctxt = {
        srv: srv,
        def_set: def_set,
        def_map: common::new_def_hash()
    };

    // FIXME: Do a parallel fold
    let fold = fold::fold({
        fold_mod: fold_mod,
        fold_nmod: fold_nmod
        with *fold::default_seq_fold(ctxt)
    });

    fold.fold_doc(fold, doc);

    ret ctxt.def_map;

    fn fold_mod(fold: fold::fold<ctxt>, doc: doc::moddoc) -> doc::moddoc {
        let doc = fold::default_seq_fold_mod(fold, doc);

        for doc.items.each {|item|
            let def_id = ast_util::local_def(item.id());
            if fold.ctxt.def_set.contains_key(def_id) {
                fold.ctxt.def_map.insert(def_id, item);
            }
        }

        ret doc;
    }

    fn fold_nmod(fold: fold::fold<ctxt>, doc: doc::nmoddoc) -> doc::nmoddoc {
        let doc = fold::default_seq_fold_nmod(fold, doc);

        for doc.fns.each {|fndoc|
            let def_id = ast_util::local_def(fndoc.id());
            if fold.ctxt.def_set.contains_key(def_id) {
                fold.ctxt.def_map.insert(def_id, doc::fntag(fndoc));
            }
        }

        ret doc;
    }
}

fn build_reexport_path_map(srv: astsrv::srv, -def_map: def_map) -> path_map {

    // This is real unfortunate. Lots of copying going on here
    let def_assoc_list = to_assoc_list(def_map);
    #debug("def_map: %?", def_assoc_list);

    let assoc_list = astsrv::exec(srv) {|ctxt|

        let def_map = from_def_assoc_list(def_assoc_list);
        let path_map = map::str_hash::<[(str,doc::itemtag)]>();

        for ctxt.exp_map.each {|exp_id, defs|
            let path = alt check ctxt.ast_map.get(exp_id) {
              ast_map::node_export(_, path) { path }
            };
          // should be a constraint on the node_export constructor
          // that guarantees path is non-empty
            let name = alt check vec::last(*path) {
              ast_map::path_name(nm) { nm }
            };
            let modpath = ast_map::path_to_str(vec::init(*path));

            let mut reexportdocs = [];
            for defs.each {|def|
                if !def.reexp { cont; }
                alt def_map.find(def.id) {
                  some(itemtag) {
                    reexportdocs += [(name, itemtag)];
                  }
                  _ {}
                }
            }

            if reexportdocs.len() > 0u {
                option::iter(path_map.find(modpath)) {|docs|
                    reexportdocs = docs + vec::filter(reexportdocs, {|x|
                        !vec::contains(docs, x)
                    });
                }
                path_map.insert(modpath, reexportdocs);
                #debug("path_map entry: %? - %?",
                       modpath, (name, reexportdocs));
            }
        }

        for find_reexport_impl_docs(ctxt, def_map).each {|elt|
            let (path, doc) = elt;
            let docs = alt path_map.find(path) {
              some(docs) { docs + [(doc)] }
              none { [doc] }
            };
            path_map.insert(path, docs);
        }

        to_assoc_list(path_map)
    };

    from_str_assoc_list(assoc_list)
}

fn find_reexport_impl_docs(
    ctxt: astsrv::ctxt,
    def_map: def_map
) -> [(str, (str, doc::itemtag))] {
    let docs = @mut [];

    for_each_reexported_impl(ctxt) {|mod_id, i|
        let path = alt ctxt.ast_map.find(mod_id) {
          some(ast_map::node_item(item, path)) {
            let path = ast_map::path_to_str(*path);
            if str::is_empty(path) {
                item.ident
            } else {
                path + "::" + item.ident
            }
          }
          _ {
            assert mod_id == ast::crate_node_id;
            ""
          }
        };
        let ident = i.ident;
        let doc = alt check def_map.find(i.did) {
          some(doc) { doc }
        };
        *docs += [(path, (ident, doc))];
    }

    ret *docs;
}

fn for_each_reexported_impl(
    ctxt: astsrv::ctxt,
    f: fn@(ast::node_id, resolve::_impl)
) {
    let visitor = @{
        visit_mod: bind visit_mod(ctxt, f, _, _, _)
        with *visit::default_simple_visitor()
    };
    let visitor = visit::mk_simple_visitor(visitor);
    visit::visit_crate(*ctxt.ast, (), visitor);

    fn visit_mod(
        ctxt: astsrv::ctxt,
        f: fn@(ast::node_id, resolve::_impl),
        m: ast::_mod,
        _sp: codemap::span,
        mod_id: ast::node_id
    ) {
        let all_impls = all_impls(m);
        alt check ctxt.impl_map.get(mod_id) {
          list::cons(impls, @list::nil) {
            for vec::each(*impls) {|i|
                // This impl is not an item in the current mod
                if !all_impls.contains_key(i.did) {
                    // Ignore external impls because I don't
                    // know what to do with them yet
                    if i.did.crate == ast::local_crate {
                        f(mod_id, *i);
                    }
                }
            }
          }
        }
    }
}

fn all_impls(m: ast::_mod) -> map::set<ast::def_id> {
    let all_impls = common::new_def_hash();
    for m.items.each {|item|
        alt item.node {
          ast::item_impl(_, _, _, _, _) {
            all_impls.insert(ast_util::local_def(item.id), ());
          }
          _ { }
        }
    }
    ret all_impls;
}

fn merge_reexports(
    doc: doc::doc,
    path_map: path_map
) -> doc::doc {

    let fold = fold::fold({
        fold_mod: fold_mod
        with *fold::default_seq_fold(path_map)
    });

    ret fold.fold_doc(fold, doc);

    fn fold_mod(fold: fold::fold<path_map>, doc: doc::moddoc) -> doc::moddoc {
        let doc = fold::default_seq_fold_mod(fold, doc);

        let is_topmod = doc.id() == ast::crate_node_id;

        // In the case of the top mod, it really doesn't have a name;
        // the name we have here is actually the crate name
        let path = if is_topmod {
            doc.path()
        } else {
            doc.path() + [doc.name()]
        };

        let new_items = get_new_items(path, fold.ctxt);
        #debug("merging into %?: %?", path, new_items);

        {
            items: (doc.items + new_items)
            with doc
        }
    }

    fn get_new_items(path: [str], path_map: path_map) -> [doc::itemtag] {
        #debug("looking for reexports in path %?", path);
        alt path_map.find(str::connect(path, "::")) {
          some(name_docs) {
            vec::foldl([], name_docs) {|v, name_doc|
                let (name, doc) = name_doc;
                v + [reexport_doc(doc, name)]
            }
          }
          none { [] }
        }
    }

    fn reexport_doc(doc: doc::itemtag, name: str) -> doc::itemtag {
        alt doc {
          doc::modtag(doc @ {item, _}) {
            doc::modtag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::nmodtag(_) { fail }
          doc::consttag(doc @ {item, _}) {
            doc::consttag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::fntag(doc @ {item, _}) {
            doc::fntag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::enumtag(doc @ {item, _}) {
            doc::enumtag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::restag(doc @ {item, _}) {
            doc::restag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::ifacetag(doc @ {item, _}) {
            doc::ifacetag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::impltag(doc @ {item, _}) {
            doc::impltag({
                item: reexport(item, name)
                with doc
            })
          }
          doc::tytag(doc @ {item, _}) {
            doc::tytag({
                item: reexport(item, name)
                with doc
            })
          }
        }
    }

    fn reexport(doc: doc::itemdoc, name: str) -> doc::itemdoc {
        {
            name: name,
            reexport: true
            with doc
        }
    }
}

#[test]
fn should_duplicate_reexported_items() {
    let source = "mod a { export b; fn b() { } } \
                  mod c { import a::b; export b; }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().mods()[1].fns()[0].name() == "b";
}

#[test]
fn should_mark_reepxorts_as_such() {
    let source = "mod a { export b; fn b() { } } \
                  mod c { import a::b; export b; }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().mods()[1].fns()[0].item.reexport == true;
}

#[test]
fn should_duplicate_reexported_impls() {
    let source = "mod a { impl b for int { fn c() { } } } \
                  mod d { import a::b; export b; }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().mods()[1].impls()[0].name() == "b";
}

#[test]
fn should_duplicate_reexported_impls_deep() {
    let source = "mod a { impl b for int { fn c() { } } } \
                  mod d { mod e { import a::b; export b; } }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().mods()[1].mods()[0].impls()[0].name() == "b";
}

#[test]
fn should_duplicate_reexported_impls_crate() {
    let source = "import a::b; export b; \
                  mod a { impl b for int { fn c() { } } }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().impls()[0].name() == "b";
}

#[test]
fn should_duplicate_reexported_native_fns() {
    let source = "native mod a { fn b(); } \
                  mod c { import a::b; export b; }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().mods()[0].fns()[0].name() == "b";
}

#[test]
fn should_duplicate_multiple_reexported_items() {
    let source = "mod a { \
                  export b; export c; \
                  fn b() { } fn c() { } \
                  } \
                  mod d { \
                  import a::b; import a::c; \
                  export b; export c; \
                  }";
    astsrv::from_str(source) {|srv|
        let doc = extract::from_srv(srv, "");
        let doc = path_pass::mk_pass().f(srv, doc);
        let doc = run(srv, doc);
        // Reexports may not be in any specific order
        let doc = sort_item_name_pass::mk_pass().f(srv, doc);
        assert doc.cratemod().mods()[1].fns()[0].name() == "b";
        assert doc.cratemod().mods()[1].fns()[1].name() == "c";
    }
}

#[test]
fn should_rename_items_reexported_with_different_names() {
    let source = "mod a { export b; fn b() { } } \
                  mod c { import x = a::b; export x; }";
    let doc = test::mk_doc(source);
    assert doc.cratemod().mods()[1].fns()[0].name() == "x";
}

#[test]
fn should_reexport_in_topmod() {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "core");
            let doc = path_pass::mk_pass().f(srv, doc);
            run(srv, doc)
        }
    }
    let source = "import option::{some, none}; \
                  import option = option::t; \
                  export option, some, none; \
                  mod option { \
                  enum t { some, none } \
                  }";
    let doc = mk_doc(source);
    assert doc.cratemod().enums()[0].name() == "option";
}

#[test]
fn should_not_reexport_multiple_times() {
    let source = "import option = option::t; \
                  export option; \
                  export option; \
                  mod option { \
                  enum t { none, some } \
                  }";
    let doc = test::mk_doc(source);
    assert vec::len(doc.cratemod().enums()) == 1u;
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            let doc = path_pass::mk_pass().f(srv, doc);
            run(srv, doc)
        }
    }
}
