export fold;
export fold_crate, fold_mod, fold_fn, fold_modlist, fold_fnlist;
export default_seq_fold;
export default_seq_fold_crate;
export default_seq_fold_fn;
export default_seq_fold_fnlist;

tag fold<T> = t<T>;

type fold_crate<T> = fn~(
    fold: fold<T>,
    doc: doc::cratedoc
) -> doc::cratedoc;

type fold_mod<T> = fn~(
    fold: fold<T>,
    doc: doc::moddoc
) -> doc::moddoc;

type fold_fn<T> = fn~(
    fold: fold<T>,
    doc: doc::fndoc
) -> doc::fndoc;

type fold_modlist<T> = fn~(
    fold: fold<T>,
    list: doc::modlist
) -> doc::modlist;

type fold_fnlist<T> = fn~(
    fold: fold<T>,
    list: doc::fnlist
) -> doc::fnlist;

type t<T> = {
    ctxt: T,
    fold_crate: fold_crate<T>,
    fold_mod: fold_mod<T>,
    fold_fn: fold_fn<T>,
    fold_modlist: fold_modlist<T>,
    fold_fnlist: fold_fnlist<T>
};


// This exists because fn types don't infer correctly as record
// initializers, but they do as function arguments
fn mk_fold<T:copy>(
    ctxt: T,
    fold_crate: fold_crate<T>,
    fold_mod: fold_mod<T>,
    fold_fn: fold_fn<T>,
    fold_modlist: fold_modlist<T>,
    fold_fnlist: fold_fnlist<T>
) -> fold<T> {
    fold({
        ctxt: ctxt,
        fold_crate: fold_crate,
        fold_mod: fold_mod,
        fold_fn: fold_fn,
        fold_modlist: fold_modlist,
        fold_fnlist: fold_fnlist
    })
}

fn default_seq_fold<T:copy>(ctxt: T) -> fold<T> {
    mk_fold(
        ctxt,
        {|f, d| default_seq_fold_crate(f, d)},
        {|f, d| default_seq_fold_mod(f, d)},
        {|f, d| default_seq_fold_fn(f, d)},
        {|f, d| default_seq_fold_modlist(f, d)},
        {|f, d| default_seq_fold_fnlist(f, d)}
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
        name: doc.name,
        mods: fold.fold_modlist(fold, doc.mods),
        fns: fold.fold_fnlist(fold, doc.fns)
    }
}

fn default_seq_fold_fn<T>(
    _fold: fold<T>,
    doc: doc::fndoc
) -> doc::fndoc {
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

#[cfg(test)]
mod tests {
    #[test]
    fn default_fold_should_produce_same_doc() {
        let source = "mod a { fn b() { } mod c { fn d() { } } }";
        let ast = parse::from_str(source);
        let doc = extract::extract(ast, "");
        let fld = default_seq_fold(());
        let folded = fld.fold_crate(fld, doc);
        assert doc == folded;
    }
}