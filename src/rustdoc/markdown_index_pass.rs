//! Build indexes as appropriate for the markdown pass

import doc::item_utils;

export mk_pass;

fn mk_pass(config: config::config) -> pass {
    {
        name: ~"markdown_index",
        f: fn~(srv: astsrv::srv, doc: doc::doc) -> doc::doc {
            run(srv, doc, config)
        }
    }
}

fn run(
    _srv: astsrv::srv,
    doc: doc::doc,
    config: config::config
) -> doc::doc {
    let fold = fold::fold({
        fold_mod: fold_mod,
        fold_nmod: fold_nmod,
        .. *fold::default_any_fold(config)
    });
    fold.fold_doc(fold, doc)
}

fn fold_mod(
    fold: fold::fold<config::config>,
    doc: doc::moddoc
) -> doc::moddoc {

    let doc = fold::default_any_fold_mod(fold, doc);

    doc::moddoc_({
        index: Some(build_mod_index(doc, fold.ctxt)),
        .. *doc
    })
}

fn fold_nmod(
    fold: fold::fold<config::config>,
    doc: doc::nmoddoc
) -> doc::nmoddoc {

    let doc = fold::default_any_fold_nmod(fold, doc);

    {
        index: Some(build_nmod_index(doc, fold.ctxt)),
        .. doc
    }
}

fn build_mod_index(
    doc: doc::moddoc,
    config: config::config
) -> doc::index {
    {
        entries: par::map(doc.items, |doc| {
            item_to_entry(doc, config)
        })
    }
}

fn build_nmod_index(
    doc: doc::nmoddoc,
    config: config::config
) -> doc::index {
    {
        entries: par::map(doc.fns, |doc| {
            item_to_entry(doc::fntag(doc), config)
        })
    }
}

fn item_to_entry(
    doc: doc::itemtag,
    config: config::config
) -> doc::index_entry {
    let link = match doc {
      doc::modtag(_) | doc::nmodtag(_)
      if config.output_style == config::doc_per_mod => {
        markdown_writer::make_filename(config, doc::itempage(doc)).to_str()
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
        str::replace(s, ~" ", ~"-")
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
}

#[test]
fn should_index_mod_contents() {
    let doc = test::mk_doc(
        config::doc_per_crate,
        ~"mod a { } fn b() { }"
    );
    assert option::get(doc.cratemod().index).entries[0] == {
        kind: ~"Module",
        name: ~"a",
        brief: None,
        link: ~"#module-a"
    };
    assert option::get(doc.cratemod().index).entries[1] == {
        kind: ~"Function",
        name: ~"b",
        brief: None,
        link: ~"#function-b"
    };
}

#[test]
fn should_index_mod_contents_multi_page() {
    let doc = test::mk_doc(
        config::doc_per_mod,
        ~"mod a { } fn b() { }"
    );
    assert option::get(doc.cratemod().index).entries[0] == {
        kind: ~"Module",
        name: ~"a",
        brief: None,
        link: ~"a.html"
    };
    assert option::get(doc.cratemod().index).entries[1] == {
        kind: ~"Function",
        name: ~"b",
        brief: None,
        link: ~"#function-b"
    };
}

#[test]
fn should_index_foreign_mod_pages() {
    let doc = test::mk_doc(
        config::doc_per_mod,
        ~"extern mod a { }"
    );
    assert option::get(doc.cratemod().index).entries[0] == {
        kind: ~"Foreign module",
        name: ~"a",
        brief: None,
        link: ~"a.html"
    };
}

#[test]
fn should_add_brief_desc_to_index() {
    let doc = test::mk_doc(
        config::doc_per_mod,
        ~"#[doc = \"test\"] mod a { }"
    );
    assert option::get(doc.cratemod().index).entries[0].brief
        == Some(~"test");
}

#[test]
fn should_index_foreign_mod_contents() {
    let doc = test::mk_doc(
        config::doc_per_crate,
        ~"extern mod a { fn b(); }"
    );
    assert option::get(doc.cratemod().nmods()[0].index).entries[0] == {
        kind: ~"Function",
        name: ~"b",
        brief: None,
        link: ~"#function-b"
    };
}

#[cfg(test)]
mod test {
    fn mk_doc(output_style: config::output_style, source: ~str) -> doc::doc {
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
