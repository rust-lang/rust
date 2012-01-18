/* rustdoc: rust -> markdown translator
 * Copyright 2011 Google Inc.
 */

#[doc = "A single operation on the document model"]
type pass = fn~(srv: astsrv::srv, doc: doc::cratedoc) -> doc::cratedoc;

#[doc = "Run a series of passes over the document"]
fn run_passes(
    srv: astsrv::srv,
    doc: doc::cratedoc,
    passes: [pass]
) -> doc::cratedoc {
    vec::foldl(doc, passes) {|doc, pass|
        pass(srv, doc)
    }
}

#[test]
fn test_run_passes() {
    fn pass1(
        _srv: astsrv::srv,
        doc: doc::cratedoc
    ) -> doc::cratedoc {
        ~{
            topmod: ~{
                name: doc.topmod.name + "two",
                mods: doc::modlist([]),
                fns: doc::fnlist([])
            }
        }
    }
    fn pass2(
        _srv: astsrv::srv,
        doc: doc::cratedoc
    ) -> doc::cratedoc {
        ~{
            topmod: ~{
                name: doc.topmod.name + "three",
                mods: doc::modlist([]),
                fns: doc::fnlist([])
            }
        }
    }
    let source = "";
    let srv = astsrv::mk_srv_from_str(source);
    let passes = [pass1, pass2];
    let doc = extract::from_srv(srv, "one");
    let doc = run_passes(srv, doc, passes);
    assert doc.topmod.name == "onetwothree";
}

fn main(argv: [str]) {

    if vec::len(argv) != 2u {
        std::io::println(#fmt("usage: %s <input>", argv[0]));
        ret;
    }

    let source_file = argv[1];
    run(source_file);
}

#[doc = "Runs rustdoc over the given file"]
fn run(source_file: str) {

    let default_name = source_file;
    let srv = astsrv::mk_srv_from_file(source_file);
    let doc = extract::from_srv(srv, default_name);
    run_passes(srv, doc, [
        attr_pass::mk_pass(),
        // FIXME: This pass should be optional
        prune_undoc_pass::mk_pass(),
        tystr_pass::mk_pass(),
        gen::mk_pass {|| std::io:: stdout()}
    ]);
}