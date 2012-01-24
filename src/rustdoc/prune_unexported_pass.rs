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
    ~{
        mods: doc::modlist(exported_mods(fold.ctxt, doc)),
        fns: doc::fnlist(exported_fns(fold.ctxt, doc)),
        consts: doc::constlist(exported_consts(fold.ctxt, doc))
        with *doc
    }
}

fn exported_mods(srv: astsrv::srv, doc: doc::moddoc) -> [doc::moddoc] {
    exported_things(
        srv, doc,
        exported_mods_from_crate,
        exported_mods_from_mod
    )
}

fn exported_fns(srv: astsrv::srv, doc: doc::moddoc) -> [doc::fndoc] {
    exported_things(
        srv, doc,
        exported_fns_from_crate,
        exported_fns_from_mod
    )
}

fn exported_consts(srv: astsrv::srv, doc: doc::moddoc) -> [doc::constdoc] {
    exported_things(
        srv, doc,
        exported_consts_from_crate,
        exported_consts_from_mod
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

fn exported_mods_from_crate(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::moddoc] {
    exported_mods_from(srv, doc, is_exported_from_crate)
}

fn exported_mods_from_mod(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::moddoc] {
    exported_mods_from(srv, doc, bind is_exported_from_mod(_, doc.id, _))
}

fn exported_fns_from_crate(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::fndoc] {
    exported_fns_from(srv, doc, is_exported_from_crate)
}

fn exported_fns_from_mod(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::fndoc] {
    exported_fns_from(srv, doc, bind is_exported_from_mod(_, doc.id, _))
}

fn exported_consts_from_crate(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::constdoc] {
    exported_consts_from(srv, doc, is_exported_from_crate)
}

fn exported_consts_from_mod(
    srv: astsrv::srv,
    doc: doc::moddoc
) -> [doc::constdoc] {
    exported_consts_from(srv, doc, bind is_exported_from_mod(_, doc.id, _))
}

fn exported_fns_from(
    srv: astsrv::srv,
    doc: doc::moddoc,
    is_exported: fn(astsrv::srv, str) -> bool
) -> [doc::fndoc] {
    vec::filter_map(*doc.fns) { |doc|
        if is_exported(srv, doc.name) {
            some(doc)
        } else {
            none
        }
    }
}

fn exported_mods_from(
    srv: astsrv::srv,
    doc: doc::moddoc,
    is_exported: fn(astsrv::srv, str) -> bool
) -> [doc::moddoc] {
    vec::filter_map(*doc.mods) { |doc|
        if is_exported(srv, doc.name) {
            some(doc)
        } else {
            none
        }
    }
}

fn exported_consts_from(
    srv: astsrv::srv,
    doc: doc::moddoc,
    is_exported: fn(astsrv::srv, str) -> bool
) -> [doc::constdoc] {
    vec::filter_map(*doc.consts) { |doc|
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
        alt ctxt.map.get(mod_id) {
          ast_map::node_item(item) {
            alt item.node {
              ast::item_mod(m) {
                ast_util::is_exported(item_name, m)
              }
            }
          }
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
    let source = "mod b { export a; fn a() { } fn b() { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::len(*doc.topmod.mods[0].fns) == 1u;
}

#[test]
fn should_prune_unexported_fns_from_top_mod() {
    let source = "export a; fn a() { } fn b() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::len(*doc.topmod.fns) == 1u;
}

#[test]
fn should_prune_unexported_modules() {
    let source = "mod a { export a; mod a { } mod b { } }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::len(*doc.topmod.mods[0].mods) == 1u;
}

#[test]
fn should_prune_unexported_modules_from_top_mod() {
    let source = "export a; mod a { } mod b { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::len(*doc.topmod.mods) == 1u;
}

#[test]
fn should_prune_unexported_consts() {
    let source = "mod a { export a; \
                  const a: bool = true; \
                  const b: bool = true; }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::len(*doc.topmod.mods[0].consts) == 1u;
}

#[test]
fn should_prune_unexported_consts_from_top_mod() {
    let source = "export a; const a: bool = true; const b: bool = true;";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = run(srv, doc);
    assert vec::len(*doc.topmod.consts) == 1u;
}
