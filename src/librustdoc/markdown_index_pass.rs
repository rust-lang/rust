//! Build indexes as appropriate for the markdown pass

use doc::ItemUtils;

export mk_pass;

fn mk_pass(config: config::Config) -> Pass {
    {
        name: ~"markdown_index",
        f: fn~(srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc {
            run(srv, doc, config)
        }
    }
}

fn run(
    _srv: astsrv::Srv,
    doc: doc::Doc,
    config: config::Config
) -> doc::Doc {
    let fold = fold::Fold({
        fold_mod: fold_mod,
        fold_nmod: fold_nmod,
        .. *fold::default_any_fold(config)
    });
    fold.fold_doc(fold, doc)
}

fn fold_mod(
    fold: fold::Fold<config::Config>,
    doc: doc::ModDoc
) -> doc::ModDoc {

    let doc = fold::default_any_fold_mod(fold, doc);

    doc::ModDoc_({
        index: Some(build_mod_index(doc, fold.ctxt)),
        .. *doc
    })
}

fn fold_nmod(
    fold: fold::Fold<config::Config>,
    doc: doc::NmodDoc
) -> doc::NmodDoc {

    let doc = fold::default_any_fold_nmod(fold, doc);

    {
        index: Some(build_nmod_index(doc, fold.ctxt)),
        .. doc
    }
}

fn build_mod_index(
    doc: doc::ModDoc,
    config: config::Config
) -> doc::Index {
    {
        entries: par::map(doc.items, |doc| {
            item_to_entry(*doc, config)
        })
    }
}

fn build_nmod_index(
    doc: doc::NmodDoc,
    config: config::Config
) -> doc::Index {
    {
        entries: par::map(doc.fns, |doc| {
            item_to_entry(doc::FnTag(*doc), config)
        })
    }
}

fn item_to_entry(
    doc: doc::ItemTag,
    config: config::Config
) -> doc::IndexEntry {
    let link = match doc {
      doc::ModTag(_) | doc::NmodTag(_)
      if config.output_style == config::DocPerMod => {
        markdown_writer::make_filename(config, doc::ItemPage(doc)).to_str()
      }
      _ => {
        ~"#" + pandoc_header_id(markdown_pass::header_text(doc))
      }
    };

    {
        kind: markdown_pass::header_kind(doc),
        name: markdown_pass::header_name(doc),
        brief: doc.brief(),
        link: link
    }
}

fn pandoc_header_id(header: ~str) -> ~str {

    // http://johnmacfarlane.net/pandoc/README.html#headers

    let header = remove_formatting(header);
    let header = remove_punctuation(header);
    let header = replace_with_hyphens(header);
    let header = convert_to_lowercase(header);
    let header = remove_up_to_first_letter(header);
    let header = maybe_use_section_id(header);
    return header;

    fn remove_formatting(s: ~str) -> ~str {
        str::replace(s, ~"`", ~"")
    }
    fn remove_punctuation(s: ~str) -> ~str {
        let s = str::replace(s, ~"<", ~"");
        let s = str::replace(s, ~">", ~"");
        let s = str::replace(s, ~"[", ~"");
        let s = str::replace(s, ~"]", ~"");
        let s = str::replace(s, ~"(", ~"");
        let s = str::replace(s, ~")", ~"");
        let s = str::replace(s, ~"@~", ~"");
        let s = str::replace(s, ~"~", ~"");
        let s = str::replace(s, ~"/", ~"");
        let s = str::replace(s, ~":", ~"");
        let s = str::replace(s, ~"&", ~"");
        let s = str::replace(s, ~"^", ~"");
        return s;
    }
    fn replace_with_hyphens(s: ~str) -> ~str {
        // Collapse sequences of whitespace to a single dash
        // XXX: Hacky implementation here that only covers
        // one or two spaces.
        let s = str::replace(s, ~"  ", ~"-");
        let s = str::replace(s, ~" ", ~"-");
        return s;
    }
    fn convert_to_lowercase(s: ~str) -> ~str { str::to_lower(s) }
    fn remove_up_to_first_letter(s: ~str) -> ~str { s }
    fn maybe_use_section_id(s: ~str) -> ~str { s }
}

#[test]
fn should_remove_punctuation_from_headers() {
    assert pandoc_header_id(~"impl foo of bar<A>") == ~"impl-foo-of-bara";
    assert pandoc_header_id(~"fn@(~[~A])") == ~"fna";
    assert pandoc_header_id(~"impl of num::num for int")
        == ~"impl-of-numnum-for-int";
    assert pandoc_header_id(~"impl of num::num for int/&")
        == ~"impl-of-numnum-for-int";
    assert pandoc_header_id(~"impl of num::num for ^int")
        == ~"impl-of-numnum-for-int";
    assert pandoc_header_id(~"impl for & condvar")
        == ~"impl-for-condvar";
}

#[test]
fn should_index_mod_contents() {
    let doc = test::mk_doc(
        config::DocPerCrate,
        ~"mod a { } fn b() { }"
    );
    assert doc.cratemod().index.get().entries[0] == {
        kind: ~"Module",
        name: ~"a",
        brief: None,
        link: ~"#module-a"
    };
    assert doc.cratemod().index.get().entries[1] == {
        kind: ~"Function",
        name: ~"b",
        brief: None,
        link: ~"#function-b"
    };
}

#[test]
fn should_index_mod_contents_multi_page() {
    let doc = test::mk_doc(
        config::DocPerMod,
        ~"mod a { } fn b() { }"
    );
    assert doc.cratemod().index.get().entries[0] == {
        kind: ~"Module",
        name: ~"a",
        brief: None,
        link: ~"a.html"
    };
    assert doc.cratemod().index.get().entries[1] == {
        kind: ~"Function",
        name: ~"b",
        brief: None,
        link: ~"#function-b"
    };
}

#[test]
fn should_index_foreign_mod_pages() {
    let doc = test::mk_doc(
        config::DocPerMod,
        ~"extern mod a { }"
    );
    assert doc.cratemod().index.get().entries[0] == {
        kind: ~"Foreign module",
        name: ~"a",
        brief: None,
        link: ~"a.html"
    };
}

#[test]
fn should_add_brief_desc_to_index() {
    let doc = test::mk_doc(
        config::DocPerMod,
        ~"#[doc = \"test\"] mod a { }"
    );
    assert doc.cratemod().index.get().entries[0].brief
        == Some(~"test");
}

#[test]
fn should_index_foreign_mod_contents() {
    let doc = test::mk_doc(
        config::DocPerCrate,
        ~"extern mod a { fn b(); }"
    );
    assert doc.cratemod().nmods()[0].index.get().entries[0] == {
        kind: ~"Function",
        name: ~"b",
        brief: None,
        link: ~"#function-b"
    };
}

#[cfg(test)]
mod test {
    #[legacy_exports];
    fn mk_doc(output_style: config::OutputStyle, source: ~str) -> doc::Doc {
        do astsrv::from_str(source) |srv| {
            let config = {
                output_style: output_style,
                .. config::default_config(&Path("whatever"))
            };
            let doc = extract::from_srv(srv, ~"");
            let doc = attr_pass::mk_pass().f(srv, doc);
            let doc = desc_to_brief_pass::mk_pass().f(srv, doc);
            let doc = path_pass::mk_pass().f(srv, doc);
            run(srv, doc, config)
        }
    }
}
