// FIXME: Random import to solve the mystery resolve bug
import std;

export fold;
export fold_crate, fold_mod, fold_fn, fold_modlist, fold_fnlist;
export default_seq_fold;
export default_seq_fold_crate;
export default_seq_fold_mod;
export default_seq_fold_fn;
export default_seq_fold_const;
export default_seq_fold_enum;
export default_seq_fold_fnlist;

enum fold<T> = t<T>;

type fold_crate<T> = fn~(fold: fold<T>, doc: doc::cratedoc) -> doc::cratedoc;
type fold_mod<T> = fn~(fold: fold<T>, doc: doc::moddoc) -> doc::moddoc;
type fold_fn<T> = fn~(fold: fold<T>, doc: doc::fndoc) -> doc::fndoc;
type fold_const<T> = fn~(fold: fold<T>, doc: doc::constdoc) -> doc::constdoc;
type fold_enum<T> = fn~(fold: fold<T>, doc: doc::enumdoc) -> doc::enumdoc;
type fold_modlist<T> = fn~(fold: fold<T>, list: doc::modlist) -> doc::modlist;
type fold_fnlist<T> = fn~(fold: fold<T>, list: doc::fnlist) -> doc::fnlist;
type fold_constlist<T> = fn~(
    fold: fold<T>, list: doc::constlist) -> doc::constlist;
type fold_enumlist<T> = fn~(
    fold: fold<T>, list: doc::enumlist) -> doc::enumlist;

type t<T> = {
    ctxt: T,
    fold_crate: fold_crate<T>,
    fold_mod: fold_mod<T>,
    fold_fn: fold_fn<T>,
    fold_const: fold_const<T>,
    fold_enum: fold_enum<T>,
    fold_modlist: fold_modlist<T>,
    fold_fnlist: fold_fnlist<T>,
    fold_constlist: fold_constlist<T>,
    fold_enumlist: fold_enumlist<T>
};


// This exists because fn types don't infer correctly as record
// initializers, but they do as function arguments
fn mk_fold<T:copy>(
    ctxt: T,
    fold_crate: fold_crate<T>,
    fold_mod: fold_mod<T>,
    fold_fn: fold_fn<T>,
    fold_const: fold_const<T>,
    fold_enum: fold_enum<T>,
    fold_modlist: fold_modlist<T>,
    fold_fnlist: fold_fnlist<T>,
    fold_constlist: fold_constlist<T>,
    fold_enumlist: fold_enumlist<T>
) -> fold<T> {
    fold({
        ctxt: ctxt,
        fold_crate: fold_crate,
        fold_mod: fold_mod,
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_modlist: fold_modlist,
        fold_fnlist: fold_fnlist,
        fold_constlist: fold_constlist,
        fold_enumlist: fold_enumlist
    })
}

fn default_seq_fold<T:copy>(ctxt: T) -> fold<T> {
    mk_fold(
        ctxt,
        {|f, d| default_seq_fold_crate(f, d)},
        {|f, d| default_seq_fold_mod(f, d)},
        {|f, d| default_seq_fold_fn(f, d)},
        {|f, d| default_seq_fold_const(f, d)},
        {|f, d| default_seq_fold_enum(f, d)},
        {|f, d| default_seq_fold_modlist(f, d)},
        {|f, d| default_seq_fold_fnlist(f, d)},
        {|f, d| default_seq_fold_constlist(f, d)},
        {|f, d| default_seq_fold_enumlist(f, d)}
    )
}

fn default_seq_fold_crate<T>(
    fold: fold<T>,
    doc: doc::cratedoc
) -> doc::cratedoc {
    ~{
        topmod: fold.fold_mod(fold, doc.topmod)
    }
}

fn default_seq_fold_mod<T>(
    fold: fold<T>,
    doc: doc::moddoc
) -> doc::moddoc {
    ~{
        mods: fold.fold_modlist(fold, doc.mods),
        fns: fold.fold_fnlist(fold, doc.fns),
        consts: fold.fold_constlist(fold, doc.consts),
        enums: fold.fold_enumlist(fold, doc.enums)
        with *doc
    }
}

fn default_seq_fold_fn<T>(
    _fold: fold<T>,
    doc: doc::fndoc
) -> doc::fndoc {
    doc
}

fn default_seq_fold_const<T>(
    _fold: fold<T>,
    doc: doc::constdoc
) -> doc::constdoc {
    doc
}

fn default_seq_fold_enum<T>(
    _fold: fold<T>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    doc
}

fn default_seq_fold_modlist<T>(
    fold: fold<T>,
    list: doc::modlist
) -> doc::modlist {
    doc::modlist(vec::map(*list) {|doc|
        fold.fold_mod(fold, doc)
    })
}

fn default_seq_fold_fnlist<T>(
    fold: fold<T>,
    list: doc::fnlist
) -> doc::fnlist {
    doc::fnlist(vec::map(*list) {|doc|
        fold.fold_fn(fold, doc)
    })
}

fn default_seq_fold_constlist<T>(
    fold: fold<T>,
    list: doc::constlist
) -> doc::constlist {
    doc::constlist(vec::map(*list) {|doc|
        fold.fold_const(fold, doc)
    })
}

fn default_seq_fold_enumlist<T>(
    fold: fold<T>,
    list: doc::enumlist
) -> doc::enumlist {
    doc::enumlist(vec::map(*list) {|doc|
        fold.fold_enum(fold, doc)
    })
}

#[test]
fn default_fold_should_produce_same_doc() {
    let source = "mod a { fn b() { } mod c { fn d() { } } }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, "");
    let fld = default_seq_fold(());
    let folded = fld.fold_crate(fld, doc);
    assert doc == folded;
}

#[test]
fn default_fold_should_produce_same_consts() {
    let source = "const a: int = 0;";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, "");
    let fld = default_seq_fold(());
    let folded = fld.fold_crate(fld, doc);
    assert doc == folded;
}

#[test]
fn default_fold_should_produce_same_enums() {
    let source = "enum a { b }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, "");
    let fld = default_seq_fold(());
    let folded = fld.fold_crate(fld, doc);
    assert doc == folded;
}