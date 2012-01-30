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
        fold_fn: fold_fn,
        fold_enum: fold_enum,
        fold_res: fold_res
        with *fold::default_seq_fold(())
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(fold: fold::fold<()>, doc: doc::moddoc) -> doc::moddoc {
    let doc = fold::default_seq_fold_mod(fold, doc);
    let (brief, desc) = modify(doc.brief, doc.desc);

    {
        brief: brief,
        desc: desc
        with doc
    }
}

fn fold_const(fold: fold::fold<()>, doc: doc::constdoc) -> doc::constdoc {
    let doc = fold::default_seq_fold_const(fold, doc);
    let (brief, desc) = modify(doc.brief, doc.desc);

    {
        brief: brief,
        desc: desc
        with doc
    }
}

fn fold_fn(fold: fold::fold<()>, doc: doc::fndoc) -> doc::fndoc {
    let doc = fold::default_seq_fold_fn(fold, doc);
    let (brief, desc) = modify(doc.brief, doc.desc);

    {
        brief: brief,
        desc: desc
        with doc
    }
}

fn fold_enum(fold: fold::fold<()>, doc: doc::enumdoc) -> doc::enumdoc {
    let doc = fold::default_seq_fold_enum(fold, doc);
    let (brief, desc) = modify(doc.brief, doc.desc);

    {
        brief: brief,
        desc: desc
        with doc
    }
}

fn fold_res(fold: fold::fold<()>, doc: doc::resdoc) -> doc::resdoc {
    let doc = fold::default_seq_fold_res(fold, doc);
    let (brief, desc) = modify(doc.brief, doc.desc);

    {
        brief: brief,
        desc: desc
        with doc
    }
}

#[test]
fn should_promote_mod_desc() {
    let source = "#[doc(desc = \"desc\")] mod m { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.mods()[0].brief == some("desc");
    assert doc.topmod.mods()[0].desc == none;
}

#[test]
fn should_promote_const_desc() {
    let source = "#[doc(desc = \"desc\")] const a: bool = true;";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.consts()[0].brief == some("desc");
    assert doc.topmod.consts()[0].desc == none;
}

#[test]
fn should_promote_fn_desc() {
    let source = "#[doc(desc = \"desc\")] fn a() { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.fns()[0].brief == some("desc");
    assert doc.topmod.fns()[0].desc == none;
}

#[test]
fn should_promote_enum_desc() {
    let source = "#[doc(desc = \"desc\")] enum a { b }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.enums()[0].brief == some("desc");
    assert doc.topmod.enums()[0].desc == none;
}

#[test]
fn should_promote_resource_desc() {
    let source = "#[doc(desc = \"desc\")] resource r(a: bool) { }";
    let srv = astsrv::mk_srv_from_str(source);
    let doc = extract::from_srv(srv, "");
    let doc = attr_pass::mk_pass()(srv, doc);
    let doc = run(srv, doc);
    assert doc.topmod.resources()[0].brief == some("desc");
    assert doc.topmod.resources()[0].desc == none;
}

fn modify(
    brief: option<str>,
    desc: option<str>
) -> (option<str>, option<str>) {

    if option::is_some(brief) || option::is_none(desc) {
        ret (brief, desc);
    }

    parse_desc(option::get(desc))
}

fn parse_desc(desc: str) -> (option<str>, option<str>) {

    const max_brief_len: uint = 120u;

    let paras = paragraphs(desc);

    if check vec::is_not_empty(paras) {
        let maybe_brief = vec::head(paras);
        if str::char_len(maybe_brief) <= max_brief_len {
            let desc_paras = vec::tail(paras);
            let desc = if vec::is_not_empty(desc_paras) {
                some(str::connect(desc_paras, "\n\n"))
            } else {
                none
            };
            (some(maybe_brief), desc)
        } else {
            (none, some(desc))
        }
    } else {
        (none, none)
    }
}

fn paragraphs(s: str) -> [str] {
    let lines = str::lines_any(s);
    let whitespace_lines = 0;
    let accum = "";
    let paras = vec::foldl([], lines) {|paras, line|
        let res = paras;

        if str::is_whitespace(line) {
            whitespace_lines += 1;
        } else {
            if whitespace_lines > 0 {
                if str::is_not_empty(accum) {
                    res += [accum];
                    accum = "";
                }
            }

            whitespace_lines = 0;

            accum = if str::is_empty(accum) {
                line
            } else {
                accum + "\n" + line
            }
        }

        res
    };

    if str::is_not_empty(accum) {
        paras + [accum]
    } else {
        paras
    }
}

#[test]
fn test_paragraphs_1() {
    let paras = paragraphs("1\n\n2");
    assert paras == ["1", "2"];
}

#[test]
fn test_paragraphs_2() {
    let paras = paragraphs("\n\n1\n1\n\n2\n\n");
    assert paras == ["1\n1", "2"];
}

#[test]
fn should_promote_short_descs() {
    let brief = none;
    let desc = some("desc");
    let (newbrief, newdesc) = modify(brief, desc);
    assert newbrief == desc;
    assert newdesc == none;
}

#[test]
fn should_not_promote_long_descs() {
    let brief = none;
    let desc = some("Warkworth Castle is a ruined medieval building
in the town of the same name in the English county of Northumberland.
The town and castle occupy a loop of the River Coquet, less than a mile
from England's north-east coast. When the castle was founded is uncertain,
but traditionally its construction has been ascribed to Prince Henry of
Scotland in the mid 12th century, although it may have been built by
King Henry II of England when he took control of England'snorthern
counties.");
    let (newbrief, _) = modify(brief, desc);
    assert newbrief == none;
}

#[test]
fn should_not_promote_descs_over_brief() {
    let brief = some("brief");
    let desc = some("desc");
    let (newbrief, newdesc) = modify(brief, desc);
    assert newbrief == brief;
    assert newdesc == desc;
}

#[test]
fn should_extract_brief_from_desc() {
    let brief = none;
    let desc = some("brief\n\ndesc");
    let (newbrief, newdesc) = modify(brief, desc);
    assert newbrief == some("brief");
    assert newdesc == some("desc");
}
