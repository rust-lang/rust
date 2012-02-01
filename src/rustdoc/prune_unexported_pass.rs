#[doc = "Prunes branches of the tree that are not exported"];

import rustc::syntax::ast;
import rustc::syntax::ast_util;
import rustc::middle::ast_map;

export mk_pass;

fn mk_pass() -> pass {
    run
}

fn run(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
    let fold = fold::fold({
        fold_mod: fold_mod
        with *fold::default_seq_fold(srv)
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(fold: fold::fold<astsrv::srv>, doc: doc::moddoc) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);
    {
        items: ~exported_items(fold.ctxt, doc)
        with doc
    }
}

fn exported_items(srv: astsrv::srv, doc: doc::moddoc) -> [doc::itemtag] {
    exported_things(
        srv, doc,
        exported_items_from_crate,
        exported_items_from_mod
    )
}

fn exported_things<T>(
    srv: astsrv::srv,
    doc: doc::moddoc,
    from_crate: fn(astsrv::srv, doc::moddoc) -> [T],
    from_mod: fn(astsrv::srv, doc::moddoc) -> [T]
) -> [T] {
    if doc.id == ast::crate_node_id {
        from_crate(srv, doc)
    } else {
        from_mod(srv, doc)
    }
}

fn exported_items_from_crate(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::itemtag] {
    exported_items_from(srv, doc, is_exported_from_crate)
}

fn exported_items_from_mod(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::itemtag] {
    exported_items_from(srv, doc, bind is_exported_from_mod(_, doc.id, _))
}

fn exported_items_from(
    srv: astsrv::srv,
    doc: doc::moddoc,
    is_exported: fn(astsrv::srv, str) -> bool
) -> [doc::itemtag] {
    vec::filter_map(*doc.items) { |itemtag|
        let itemtag = alt itemtag {
          doc::enumtag(enumdoc) {
            // Also need to check variant exportedness
            doc::enumtag({
                variants: exported_variants_from(srv, enumdoc, is_exported)
                with enumdoc
            })
          }
          _ { itemtag }
        };
        if is_exported(srv, itemtag.name()) {
            some(itemtag)
        } else {
            none
        }
    }
}

fn exported_variants_from(
    srv: astsrv::srv,
    doc: doc::enumdoc,
    is_exported: fn(astsrv::srv, str) -> bool
) -> [doc::variantdoc] {
    vec::filter_map(doc.variants) { |doc|
        if is_exported(srv, doc.name) {
            some(doc)
        } else {
            none
        }
    }
}

fn is_exported_from_mod(
    srv: astsrv::srv,
    mod_id: doc::ast_id,
    item_name: str
) -> bool {
    astsrv::exec(srv) {|ctxt|
        alt ctxt.ast_map.get(mod_id) {
          ast_map::node_item(item) {
            alt item.node {
              ast::item_mod(m) {
                ast_util::is_exported(item_name, m)
              }
              _ {
                fail "is_exported_from_mod: not a mod";
              }
            }
          }
          _ { fail "is_exported_from_mod: not an item"; }
        }
    }
}

fn is_exported_from_crate(
    srv: astsrv::srv,
    item_name: str
) -> bool {
    astsrv::exec(srv) {|ctxt|
        ast_util::is_exported(item_name, ctxt.ast.node.module)
    }
}

#[test]
fn should_prune_unexported_fns() {
    let doc = test::mk_doc("mod b { export a; fn a() { } fn b() { } }");
    assert vec::len(doc.topmod.mods()[0].fns()) == 1u;
}

#[test]
fn should_prune_unexported_fns_from_top_mod() {
    let doc = test::mk_doc("export a; fn a() { } fn b() { }");
    assert vec::len(doc.topmod.fns()) == 1u;
}

#[test]
fn should_prune_unexported_modules() {
    let doc = test::mk_doc("mod a { export a; mod a { } mod b { } }");
    assert vec::len(doc.topmod.mods()[0].mods()) == 1u;
}

#[test]
fn should_prune_unexported_modules_from_top_mod() {
    let doc = test::mk_doc("export a; mod a { } mod b { }");
    assert vec::len(doc.topmod.mods()) == 1u;
}

#[test]
fn should_prune_unexported_consts() {
    let doc = test::mk_doc(
        "mod a { export a; \
         const a: bool = true; \
         const b: bool = true; }");
    assert vec::len(doc.topmod.mods()[0].consts()) == 1u;
}

#[test]
fn should_prune_unexported_consts_from_top_mod() {
    let doc = test::mk_doc(
        "export a; const a: bool = true; const b: bool = true;");
    assert vec::len(doc.topmod.consts()) == 1u;
}

#[test]
fn should_prune_unexported_enums_from_top_mod() {
    let doc = test::mk_doc("export a; mod a { } enum b { c }");
    assert vec::len(doc.topmod.enums()) == 0u;
}

#[test]
fn should_prune_unexported_enums() {
    let doc = test::mk_doc("mod a { export a; mod a { } enum b { c } }");
    assert vec::len(doc.topmod.mods()[0].enums()) == 0u;
}

#[test]
fn should_prune_unexported_variants_from_top_mod() {
    let doc = test::mk_doc("export b::{}; enum b { c }");
    assert vec::len(doc.topmod.enums()[0].variants) == 0u;
}

#[test]
fn should_prune_unexported_variants() {
    let doc = test::mk_doc("mod a { export b::{}; enum b { c } }");
    assert vec::len(doc.topmod.mods()[0].enums()[0].variants) == 0u;
}

#[test]
fn should_prune_unexported_resources_from_top_mod() {
    let doc = test::mk_doc("export a; mod a { } resource r(a: bool) { }");
    assert vec::is_empty(doc.topmod.resources());
}

#[test]
fn should_prune_unexported_resources() {
    let doc = test::mk_doc(
        "mod a { export a; mod a { } resource r(a: bool) { } }");
    assert vec::is_empty(doc.topmod.mods()[0].resources());
}

#[test]
fn should_prune_unexported_ifaces_from_top_mod() {
    let doc = test::mk_doc("export a; mod a { } iface b { fn c(); }");
    assert vec::is_empty(doc.topmod.ifaces());
}

#[test]
fn should_prune_unexported_impls_from_top_mod() {
    let doc = test::mk_doc(
        "export a; mod a { } impl b for int { fn c() { } }");
    assert vec::is_empty(doc.topmod.impls())
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        let srv = astsrv::mk_srv_from_str(source);
        let doc = extract::from_srv(srv, "");
        run(srv, doc)
    }
}