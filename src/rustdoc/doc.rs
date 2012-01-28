#[doc = "The document model"];

type ast_id = int;

type cratedoc = ~{
    topmod: moddoc,
};

enum itemtag {
    modtag(moddoc),
    consttag(constdoc),
    fntag(fndoc),
    enumtag(enumdoc),
    restag(resdoc)
}

type moddoc = ~{
    id: ast_id,
    name: str,
    path: [str],
    brief: option<str>,
    desc: option<str>,
    items: [itemtag],
    mods: modlist,
    fns: fnlist,
    consts: constlist
};

type constdoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    ty: option<str>
};

type fndoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    return: retdoc,
    failure: option<str>,
    sig: option<str>
};

type argdoc = ~{
    name: str,
    desc: option<str>,
    ty: option<str>
};

type retdoc = {
    desc: option<str>,
    ty: option<str>
};

type enumdoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    variants: [variantdoc]
};

type variantdoc = ~{
    name: str,
    desc: option<str>,
    sig: option<str>
};

type resdoc = ~{
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    sig: option<str>
};

// Just to break the structural recursive types
enum modlist = [moddoc];
enum constlist = [constdoc];
enum fnlist = [fndoc];

impl util for moddoc {

    fn enums() -> [enumdoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              enumtag(enumdoc) { some(enumdoc) }
              _ { none }
            }
        }
    }

    fn resources() -> [resdoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              restag(resdoc) { some(resdoc) }
              _ { none }
            }
        }
    }
}