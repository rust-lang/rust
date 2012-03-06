#[doc = "Build indexes as appropriate for the markdown pass"];

export mk_pass;

fn mk_pass(config: config::config) -> pass {
    {
        name: "markdown_index",
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
        fold_mod: fold_mod
            with *fold::default_any_fold(config)
    });
    fold.fold_doc(fold, doc)
}

fn fold_mod(
    fold: fold::fold<config::config>,
    doc: doc::moddoc
) -> doc::moddoc {

    let doc = fold::default_any_fold_mod(fold, doc);

    {
        index: some(build_index(doc, fold.ctxt))
        with doc
    }
}

fn build_index(
    doc: doc::moddoc,
    config: config::config
) -> doc::index {
    {
        entries: par::anymap(doc.items) {|item|
            item_to_entry(item, config)
        }
    }
}

fn item_to_entry(
    doc: doc::itemtag,
    config: config::config
) -> doc::index_entry {
    let link = alt doc {
      doc::modtag(_) if config.output_style == config::doc_per_mod {
        markdown_writer::make_filename(config, doc::itempage(doc))
      }
      _ {
        "#" + pandoc_header_id(markdown_pass::header_text(doc))
      }
    };

    {
        kind: markdown_pass::header_kind(doc),
        name: markdown_pass::header_name(doc),
        link: link
    }
}

fn pandoc_header_id(header: str) -> str {

    // http://johnmacfarlane.net/pandoc/README.html#headers

    let header = remove_formatting(header);
    let header = remove_punctuation(header);
    let header = replace_with_hyphens(header);
    let header = convert_to_lowercase(header);
    let header = remove_up_to_first_letter(header);
    let header = maybe_use_section_id(header);
    ret header;

    fn remove_formatting(s: str) -> str { s }
    fn remove_punctuation(s: str) -> str {
        str::replace(s, "`", "")
    }
    fn replace_with_hyphens(s: str) -> str {
        str::replace(s, " ", "-")
    }
    fn convert_to_lowercase(s: str) -> str { str::to_lower(s) }
    fn remove_up_to_first_letter(s: str) -> str { s }
    fn maybe_use_section_id(s: str) -> str { s }
}

#[test]
fn should_index_mod_contents() {
    let doc = test::mk_doc(
        config::doc_per_crate,
        "mod a { } fn b() { }"
    );
    assert option::get(doc.cratemod().index).entries[0] == {
        kind: "Module",
        name: "a",
        link: "#module-a"
    };
    assert option::get(doc.cratemod().index).entries[1] == {
        kind: "Function",
        name: "b",
        link: "#function-b"
    };
}

#[test]
fn should_index_mod_contents_multi_page() {
    let doc = test::mk_doc(
        config::doc_per_mod,
        "mod a { } fn b() { }"
    );
    assert option::get(doc.cratemod().index).entries[0] == {
        kind: "Module",
        name: "a",
        link: "a.html"
    };
    assert option::get(doc.cratemod().index).entries[1] == {
        kind: "Function",
        name: "b",
        link: "#function-b"
    };
}

#[cfg(test)]
mod test {
    fn mk_doc(output_style: config::output_style, source: str) -> doc::doc {
        astsrv::from_str(source) {|srv|
            let config = {
                output_style: output_style
                with config::default_config("whatever")
            };
            let doc = extract::from_srv(srv, "");
            let doc = path_pass::mk_pass().f(srv, doc);
            run(srv, doc, config)
        }
    }
}
