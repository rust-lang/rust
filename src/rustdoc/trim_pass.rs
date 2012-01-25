#[doc = "

Pulls a brief description out of a long description.

If the first paragraph of a long description is short enough then it
is interpreted as the brief description.

"];

export mk_pass;

fn mk_pass() -> pass {
    run
}

fn run(
    _srv: astsrv::srv,
    doc: doc::cratedoc
) -> doc::cratedoc {
    let fold = fold::fold({
        fold_mod: fold_mod,
        fold_const: fold_const,
        fold_fn: fold_fn
        with *fold::default_seq_fold(())
    });
    fold.fold_crate(fold, doc)
}

fn trimopt(s: option<str>) -> option<str> {
    option::map(s, {|s| str::trim(s) })
}

fn fold_mod(fold: fold::fold<()>, doc: doc::moddoc) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);

    ~{
        brief: trimopt(doc.brief),
        desc: trimopt(doc.desc)
        with *doc
    }
}

#[test]
fn should_trim_mod() {
    let source = "#[doc(brief = \"\nbrief\n\", \
                  desc = \"\ndesc\n\")] \
                  mod m { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.mods[0].brief == some("brief");
    assert doc.topmod.mods[0].desc == some("desc");
}

fn fold_const(fold: fold::fold<()>, doc: doc::constdoc) -> doc::constdoc {
    let doc = fold::default_seq_fold_const(fold, doc);

    ~{
        brief: trimopt(doc.brief),
        desc: trimopt(doc.desc)
        with *doc
    }
}

#[test]
fn should_trim_const() {
    let source = "#[doc(brief = \"\nbrief\n\", \
                  desc = \"\ndesc\n\")] \
                  const a: bool = true;";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.consts[0].brief == some("brief");
    assert doc.topmod.consts[0].desc == some("desc");
}

fn fold_fn(fold: fold::fold<()>, doc: doc::fndoc) -> doc::fndoc {
    let doc = fold::default_seq_fold_fn(fold, doc);

    ~{
        brief: trimopt(doc.brief),
        desc: trimopt(doc.desc),
        args: vec::map(doc.args) {|doc|
            ~{
                desc: trimopt(doc.desc)
                with *doc
            }
        },
        return: {
            desc: trimopt(doc.return.desc)
            with doc.return
        },
        failure: trimopt(doc.failure)
        with *doc
    }
}

#[test]
fn should_trim_fn() {
    let source = "#[doc(brief = \"\nbrief\n\", \
                  desc = \"\ndesc\n\")] \
                  fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.fns[0].brief == some("brief");
    assert doc.topmod.fns[0].desc == some("desc");
}

#[test]
fn should_trim_args() {
    let source = "#[doc(args(a = \"\na\n\"))] fn a(a: int) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.fns[0].args[0].desc == some("a");
}

#[test]
fn should_trim_ret() {
    let source = "#[doc(return = \"\na\n\")] fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.fns[0].return.desc == some("a");
}

#[test]
fn should_trim_failure_conditions() {
    let source = "#[doc(failure = \"\na\n\")] fn a() -> int { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.fns[0].failure == some("a");
}