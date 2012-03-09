#[doc = "

Pulls a brief description out of a long description.

If the first paragraph of a long description is short enough then it
is interpreted as the brief description.

"];

export mk_pass;

fn mk_pass() -> pass {
    text_pass::mk_pass("trim", {|s| str::trim(s)})
}

#[test]
fn should_trim_mod() {
    let doc = test::mk_doc("#[doc(brief = \"\nbrief\n\", \
                            desc = \"\ndesc\n\")] \
                            mod m { }");
    assert doc.cratemod().mods()[0].brief() == some("brief");
    assert doc.cratemod().mods()[0].desc() == some("desc");
}

#[test]
fn should_trim_const() {
    let doc = test::mk_doc("#[doc(brief = \"\nbrief\n\", \
                            desc = \"\ndesc\n\")] \
                            const a: bool = true;");
    assert doc.cratemod().consts()[0].brief() == some("brief");
    assert doc.cratemod().consts()[0].desc() == some("desc");
}

#[test]
fn should_trim_fn() {
    let doc = test::mk_doc("#[doc(brief = \"\nbrief\n\", \
                            desc = \"\ndesc\n\")] \
                            fn a() { }");
    assert doc.cratemod().fns()[0].brief() == some("brief");
    assert doc.cratemod().fns()[0].desc() == some("desc");
}

#[test]
fn should_trim_args() {
    let doc = test::mk_doc("#[doc(args(a = \"\na\n\"))] fn a(a: int) { }");
    assert doc.cratemod().fns()[0].args[0].desc == some("a");
}

#[test]
fn should_trim_ret() {
    let doc = test::mk_doc("#[doc(return = \"\na\n\")] fn a() -> int { }");
    assert doc.cratemod().fns()[0].return.desc == some("a");
}

#[test]
fn should_trim_failure_conditions() {
    let doc = test::mk_doc("#[doc(failure = \"\na\n\")] fn a() -> int { }");
    assert doc.cratemod().fns()[0].failure == some("a");
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            let doc = attr_pass::mk_pass().f(srv, doc);
            mk_pass().f(srv, doc)
        }
    }
}