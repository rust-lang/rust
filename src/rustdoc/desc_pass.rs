#[doc = "Generic pass for performing an operation on all descriptions"];

export mk_pass;

fn mk_pass(op: fn~(str) -> str) -> pass {
    fn~(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
        run(srv, doc, op)
    }
}

type op = fn~(str) -> str;

fn run(
    _srv: astsrv::srv,
    doc: doc::cratedoc,
    op: op
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_mod: fold_mod,
        fold_const: fold_const,
        fold_fn: fold_fn
        with *fold::default_seq_fold(op)
    });
    fold.fold_crate(fold, doc)
}

fn maybe_apply_op(op: op, s: option<str>) -> option<str> {
    option::map(s) {|s| op(s) }
}

fn fold_mod(fold: fold::fold<op>, doc: doc::moddoc) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);

    ~{
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc)
        with *doc
    }
}

fn fold_const(fold: fold::fold<op>, doc: doc::constdoc) -> doc::constdoc {
    let doc = fold::default_seq_fold_const(fold, doc);

    ~{
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc)
        with *doc
    }
}

fn fold_fn(fold: fold::fold<op>, doc: doc::fndoc) -> doc::fndoc {
    let doc = fold::default_seq_fold_fn(fold, doc);

    ~{
        brief: maybe_apply_op(fold.ctxt, doc.brief),
        desc: maybe_apply_op(fold.ctxt, doc.desc),
        args: vec::map(doc.args) {|doc|
            ~{
                desc: maybe_apply_op(fold.ctxt, doc.desc)
                with *doc
            }
        },
        return: {
            desc: maybe_apply_op(fold.ctxt, doc.return.desc)
            with doc.return
        },
        failure: maybe_apply_op(fold.ctxt, doc.failure)
        with *doc
    }
}
