#[doc = "Finds docs for reexported items and duplicates them"];

import std::map;
import rustc::syntax::ast;
import rustc::syntax::ast_util;
import rustc::util::common;

export mk_pass;

fn mk_pass() -> pass {
    run
}

type def_set = map::set<ast::def_id>;
type def_map = map::hashmap<ast::def_id, doc::itemtag>;
type path_map = map::hashmap<str, [(str, doc::itemtag)]>;

fn run(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {

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

    let vec = [];
    map.items {|k, v|
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
    from_assoc_list(list, bind map::new_str_hash())
}

fn build_reexport_def_set(srv: astsrv::srv) -> def_set {
    let assoc_list = astsrv::exec(srv) {|ctxt|
        let def_set = common::new_def_hash();
        ctxt.exp_map.items {|_path, defs|
            for def in *defs {
                let def_id = ast_util::def_id_of_def(def);
                def_set.insert(def_id, ());
            }
        }
        to_assoc_list(def_set)
    };

    from_def_assoc_list(assoc_list)
}

fn build_reexport_def_map(
    srv: astsrv::srv,
    doc: doc::cratedoc,
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
        fold_mod: fold_mod
        with *fold::default_seq_fold(ctxt)
    });

    fold.fold_crate(fold, doc);

    ret ctxt.def_map;

    fn fold_mod(fold: fold::fold<ctxt>, doc: doc::moddoc) -> doc::moddoc {
        let doc = fold::default_seq_fold_mod(fold, doc);

        for item in *doc.items {
            let def_id = ast_util::local_def(item.id());
            if fold.ctxt.def_set.contains_key(def_id) {
                fold.ctxt.def_map.insert(def_id, item);
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
        let path_map = map::new_str_hash();

        ctxt.exp_map.items {|path, defs|

            let path = str::split_str(path, "::");
            let modpath = str::connect(vec::init(path), "::");
            let name = option::get(vec::last(path));

            let reexportdocs = [];

            for def in *defs {
                let def_id = ast_util::def_id_of_def(def);
                alt def_map.find(def_id) {
                  some(itemtag) {
                    reexportdocs += [(name, itemtag)];
                  }
                  none { }
                }
            }

            if vec::is_not_empty(reexportdocs) {
                let prevdocs = alt path_map.find(modpath) {
                  some(docs) { docs }
                  none { [] }
                };
                let reexportdocs = prevdocs + reexportdocs;
                path_map.insert(modpath, reexportdocs);
                #debug("path_map entry: %? - %?",
                       modpath, (name, reexportdocs));
            }
        }

        to_assoc_list(path_map)
    };

    from_str_assoc_list(assoc_list)
}

fn merge_reexports(
    doc: doc::cratedoc,
    path_map: path_map
) -> doc::cratedoc {

    let fold = fold::fold({
        fold_mod: fold_mod
        with *fold::default_seq_fold(path_map)
    });

    ret fold.fold_crate(fold, doc);

    fn fold_mod(fold: fold::fold<path_map>, doc: doc::moddoc) -> doc::moddoc {
        let doc = fold::default_seq_fold_mod(fold, doc);

        let path = doc.path() + [doc.name()];
        let new_items = get_new_items(path, fold.ctxt);
        #debug("merging into %?: %?", path, new_items);

        {
            items: ~(*doc.items + new_items)
            with doc
        }
    }

    fn get_new_items(path: [str], path_map: path_map) -> [doc::itemtag] {
        #debug("looking for reexports in path %?", path);
        alt path_map.find(str::connect(path, "::")) {
          some(name_docs) {
            vec::foldl([], name_docs) {|v, name_doc|
                let (name, doc) = name_doc;
                v + [rename_doc(doc, name)]
            }
          }
          none { [] }
        }
    }

    fn rename_doc(doc: doc::itemtag, name: str) -> doc::itemtag {
        alt doc {
          doc::modtag(doc @ {item, _}) {
            doc::modtag({
                item: rename(item, name)
                with doc
            })
          }
          doc::consttag(doc @ {item, _}) {
            doc::consttag({
                item: rename(item, name)
                with doc
            })
          }
          doc::fntag(doc @ {item, _}) {
            doc::fntag({
                item: rename(item, name)
                with doc
            })
          }
          doc::enumtag(doc @ {item, _}) {
            doc::enumtag({
                item: rename(item, name)
                with doc
            })
          }
          doc::restag(doc @ {item, _}) {
            doc::restag({
                item: rename(item, name)
                with doc
            })
          }
          doc::ifacetag(doc @ {item, _}) {
            doc::ifacetag({
                item: rename(item, name)
                with doc
            })
          }
          doc::impltag(doc @ {item, _}) {
            doc::impltag({
                item: rename(item, name)
                with doc
            })
          }
          doc::tytag(doc @ {item, _}) {
            doc::tytag({
                item: rename(item, name)
                with doc
            })
          }
        }
    }

    fn rename(doc: doc::itemdoc, name: str) -> doc::itemdoc {
        {
            name: name
            with doc
        }
    }
}

#[test]
fn should_duplicate_reexported_items() {
    let source = "mod a { export b; fn b() { } } \
                  mod c { import a::b; export b; }";
    let doc = test::mk_doc(source);
    assert doc.topmod.mods()[1].fns()[0].name() == "b";
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
        let doc = path_pass::mk_pass()(srv, doc);
        let doc = run(srv, doc);
        // Reexports may not be in any specific order
        let doc = sort_item_name_pass::mk_pass()(srv, doc);
        assert doc.topmod.mods()[1].fns()[0].name() == "b";
        assert doc.topmod.mods()[1].fns()[1].name() == "c";
    }
}

#[test]
fn should_rename_items_reexported_with_different_names() {
    let source = "mod a { export b; fn b() { } } \
                  mod c { import x = a::b; export x; }";
    let doc = test::mk_doc(source);
    assert doc.topmod.mods()[1].fns()[0].name() == "x";
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            let doc = path_pass::mk_pass()(srv, doc);
            run(srv, doc)
        }
    }
}
