#[doc = "The document model"];

type ast_id = int;

type cratedoc = {
    topmod: moddoc,
};

enum itemtag {
    modtag(moddoc),
    consttag(constdoc),
    fntag(fndoc),
    enumtag(enumdoc),
    restag(resdoc),
    ifacetag(ifacedoc)
}

type moddoc = {
    id: ast_id,
    name: str,
    path: [str],
    brief: option<str>,
    desc: option<str>,
    // This box exists to break the structural recursion
    items: ~[itemtag]
};

type constdoc = {
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    ty: option<str>
};

type fndoc = {
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    return: retdoc,
    failure: option<str>,
    sig: option<str>
};

type argdoc = {
    name: str,
    desc: option<str>,
    ty: option<str>
};

type retdoc = {
    desc: option<str>,
    ty: option<str>
};

type enumdoc = {
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    variants: [variantdoc]
};

type variantdoc = {
    name: str,
    desc: option<str>,
    sig: option<str>
};

type resdoc = {
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    sig: option<str>
};

type ifacedoc = {
    id: ast_id,
    name: str,
    brief: option<str>,
    desc: option<str>,
    methods: [methoddoc]
};

type methoddoc = {
    name: str,
    brief: option<str>,
    desc: option<str>,
    args: [argdoc],
    return: retdoc,
    failure: option<str>,
    sig: option<str>
};

impl util for moddoc {

    fn mods() -> [moddoc] {
        vec::filter_map(*self.items) {|itemtag|
            alt itemtag {
              modtag(moddoc) { some(moddoc) }
              _ { none }
            }
        }
    }

    fn fns() -> [fndoc] {
        vec::filter_map(*self.items) {|itemtag|
            alt itemtag {
              fntag(fndoc) { some(fndoc) }
              _ { none }
            }
        }
    }

    fn consts() -> [constdoc] {
        vec::filter_map(*self.items) {|itemtag|
            alt itemtag {
              consttag(constdoc) { some(constdoc) }
              _ { none }
            }
        }
    }

    fn enums() -> [enumdoc] {
        vec::filter_map(*self.items) {|itemtag|
            alt itemtag {
              enumtag(enumdoc) { some(enumdoc) }
              _ { none }
            }
        }
    }

    fn resources() -> [resdoc] {
        vec::filter_map(*self.items) {|itemtag|
            alt itemtag {
              restag(resdoc) { some(resdoc) }
              _ { none }
            }
        }
    }

    fn ifaces() -> [ifacedoc] {
        vec::filter_map(*self.items) {|itemtag|
            alt itemtag {
              ifacetag(ifacedoc) { some(ifacedoc) }
              _ { none }
            }
        }
    }
}

impl util for itemtag {
    fn name() -> str {
        alt self {
          doc::modtag({name, _}) { name }
          doc::fntag({name, _}) { name }
          doc::consttag({name, _}) { name }
          doc::enumtag({name, _}) { name }
          doc::restag({name, _}) { name }
          doc::ifacetag({name, _}) { name }
        }
    }
}
