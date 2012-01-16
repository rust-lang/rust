export fold;
export fold_crate, fold_mod, fold_fn, fold_modlist, fold_fnlist;
export default_seq_fold;

tag fold = t;

type fold_crate = fn~(fold: fold, doc: doc::cratedoc) -> doc::cratedoc;
type fold_mod = fn~(fold: fold, doc: doc::moddoc) -> doc::moddoc;
type fold_fn = fn~(fold: fold, doc: doc::fndoc) -> doc::fndoc;
type fold_modlist = fn~(fold: fold, list: doc::modlist) -> doc::modlist;
type fold_fnlist = fn~(fold: fold, list: doc::fnlist) -> doc::fnlist;

type t = {
    fold_crate: fold_crate,
    fold_mod: fold_mod,
    fold_fn: fold_fn,
    fold_modlist: fold_modlist,
    fold_fnlist: fold_fnlist
};


// This exists because fn types don't infer correctly as record
// initializers, but they do as function arguments
fn mk_fold(
    fold_crate: fold_crate,
    fold_mod: fold_mod,
    fold_fn: fold_fn,
    fold_modlist: fold_modlist,
    fold_fnlist: fold_fnlist
) -> fold {
    fold({
        fold_crate: fold_crate,
        fold_mod: fold_mod,
        fold_fn: fold_fn,
        fold_modlist: fold_modlist,
        fold_fnlist: fold_fnlist
    })
}

fn default_seq_fold() -> fold {
    mk_fold(
        default_seq_fold_crate,
        default_seq_fold_mod,
        default_seq_fold_fn,
        default_seq_fold_modlist,
        default_seq_fold_fnlist
    )
}

fn default_seq_fold_crate(
    fold: fold,
    doc: doc::cratedoc
) -> doc::cratedoc {
    ~{
        topmod: fold.fold_mod(fold, doc.topmod)
    }
}

fn default_seq_fold_mod(
    fold: fold,
    doc: doc::moddoc
) -> doc::moddoc {
    ~{
        name: doc.name,
        mods: fold.fold_modlist(fold, doc.mods),
        fns: fold.fold_fnlist(fold, doc.fns)
    }
}

fn default_seq_fold_fn(
    _fold: fold,
    doc: doc::fndoc
) -> doc::fndoc {
    doc
}

fn default_seq_fold_modlist(
    fold: fold,
    list: doc::modlist
) -> doc::modlist {
    doc::modlist(vec::map(*list) {|doc|
        fold.fold_mod(fold, doc)
    })
}

fn default_seq_fold_fnlist(
    fold: fold,
    list: doc::fnlist
) -> doc::fnlist {
    doc::fnlist(vec::map(*list) {|doc|
        fold.fold_fn(fold, doc)
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn default_fold_should_produce_same_doc() {
        let source = "mod a { fn b() { } mod c { fn d() { } } }";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "");
        let fld = default_seq_fold();
        let folded = fld.fold_crate(fld, doc);
        assert doc == folded;
    }
}