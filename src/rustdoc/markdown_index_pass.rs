#[doc = "Build indexes as appropriate for the markdown pass"];

export mk_pass;

fn mk_pass() -> pass {
    {
        name: "markdown_index",
        f: run
    }
}

fn run(_srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
    let fold = fold::fold({
        fold_mod: fold_mod
            with *fold::default_any_fold(())
    });
    fold.fold_crate(fold, doc)
}

fn fold_mod(fold: fold::fold<()>, doc: doc::moddoc) -> doc::moddoc {

    let doc = fold::default_any_fold_mod(fold, doc);

    {
        index: some(build_index(doc))
        with doc
    }
}

fn build_index(doc: doc::moddoc) -> doc::index {
    {
        entries: par::anymap(doc.items, item_to_entry)
    }
}

fn item_to_entry(doc: doc::itemtag) -> doc::index_entry {
    {
        kind: markdown_pass::header_kind(doc),
        name: markdown_pass::header_name(doc),
        link: pandoc_header_id(markdown_pass::header_text(doc))
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
    let doc = test::mk_doc("mod a { } fn b() { }");
    assert option::get(doc.topmod.index).entries[0] == {
        kind: "Module",
        name: "a",
        link: "module-a"
    };
    assert option::get(doc.topmod.index).entries[1] == {
        kind: "Function",
        name: "b",
        link: "function-b"
    };
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            let doc = path_pass::mk_pass().f(srv, doc);
            run(srv, doc)
        }
    }
}
