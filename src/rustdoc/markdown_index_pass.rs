#[doc = "Build indexes as appropriate for the markdown pass"];

export mk_pass;

fn mk_pass() -> pass {
    {
        name: "markdown_index",
        f: run
    }
}

fn run(_srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc {
    doc
}

#[test]
fn should_index_mod_contents() {
    
}

#[cfg(test)]
mod test {
    fn mk_doc(source: str) -> doc::cratedoc {
        astsrv::from_str(source) {|srv|
            let doc = extract::from_srv(srv, "");
            run(srv, doc);
        }
    }
}