#[doc = "The document model"];

type ast_id = int;

type doc = {
    pages: [page]
};

enum page {
    cratepage(cratedoc),
    itempage(itemtag)
}

#[doc = "
Most rustdocs can be parsed into 'sections' according to their markdown
headers
"]
type section = {
    header: str,
    body: str
};

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
type cratedoc = {
    topmod: moddoc,
};

enum itemtag {
    modtag(moddoc),
    nmodtag(nmoddoc),
    consttag(constdoc),
    fntag(fndoc),
    enumtag(enumdoc),
    restag(resdoc),
    ifacetag(ifacedoc),
    impltag(impldoc),
    tytag(tydoc)
}

type itemdoc = {
    id: ast_id,
    name: str,
    path: [str],
    brief: option<str>,
    desc: option<str>,
    sections: [section],
    // Indicates that this node is a reexport of a different item
    reexport: bool
};

type simpleitemdoc = {
    item: itemdoc,
    sig: option<str>
};

type moddoc = {
    item: itemdoc,
    items: [itemtag],
    index: option<index>
};

type nmoddoc = {
    item: itemdoc,
    fns: [fndoc],
    index: option<index>
};

type constdoc = simpleitemdoc;

type fndoc = simpleitemdoc;

type enumdoc = {
    item: itemdoc,
    variants: [variantdoc]
};

type variantdoc = {
    name: str,
    desc: option<str>,
    sig: option<str>
};

type resdoc = simpleitemdoc;

type ifacedoc = {
    item: itemdoc,
    methods: [methoddoc]
};

type methoddoc = {
    name: str,
    brief: option<str>,
    desc: option<str>,
    sections: [section],
    sig: option<str>
};

type impldoc = {
    item: itemdoc,
    iface_ty: option<str>,
    self_ty: option<str>,
    methods: [methoddoc]
};

type tydoc = simpleitemdoc;

type index = {
    entries: [index_entry]
};

#[doc = "

A single entry in an index

Fields:

* kind - The type of thing being indexed, e.g. 'Module'
* name - The name of the thing
* brief - The brief description
* link - A format-specific string representing the link target

"]
type index_entry = {
    kind: str,
    name: str,
    brief: option<str>,
    link: str
};

impl util for doc {
    fn cratedoc() -> cratedoc {
        option::get(vec::foldl(none, self.pages) {|_m, page|
            alt page {
              doc::cratepage(doc) { some(doc) }
              _ { none }
            }
        })
    }

    fn cratemod() -> moddoc {
        self.cratedoc().topmod
    }
}

#[doc = "Some helper methods on moddoc, mostly for testing"]
impl util for moddoc {

    fn mods() -> [moddoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              modtag(moddoc) { some(moddoc) }
              _ { none }
            }
        }
    }

    fn nmods() -> [nmoddoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              nmodtag(nmoddoc) { some(nmoddoc) }
              _ { none }
            }
        }
    }

    fn fns() -> [fndoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              fntag(fndoc) { some(fndoc) }
              _ { none }
            }
        }
    }

    fn consts() -> [constdoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              consttag(constdoc) { some(constdoc) }
              _ { none }
            }
        }
    }

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

    fn ifaces() -> [ifacedoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              ifacetag(ifacedoc) { some(ifacedoc) }
              _ { none }
            }
        }
    }

    fn impls() -> [impldoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              impltag(impldoc) { some(impldoc) }
              _ { none }
            }
        }
    }

    fn types() -> [tydoc] {
        vec::filter_map(self.items) {|itemtag|
            alt itemtag {
              tytag(tydoc) { some(tydoc) }
              _ { none }
            }
        }
    }
}

impl util for [page] {

    fn mods() -> [moddoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(modtag(moddoc)) { some(moddoc) }
              _ { none }
            }
        }
    }

    fn nmods() -> [nmoddoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(nmodtag(nmoddoc)) { some(nmoddoc) }
              _ { none }
            }
        }
    }

    fn fns() -> [fndoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(fntag(fndoc)) { some(fndoc) }
              _ { none }
            }
        }
    }

    fn consts() -> [constdoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(consttag(constdoc)) { some(constdoc) }
              _ { none }
            }
        }
    }

    fn enums() -> [enumdoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(enumtag(enumdoc)) { some(enumdoc) }
              _ { none }
            }
        }
    }

    fn resources() -> [resdoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(restag(resdoc)) { some(resdoc) }
              _ { none }
            }
        }
    }

    fn ifaces() -> [ifacedoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(ifacetag(ifacedoc)) { some(ifacedoc) }
              _ { none }
            }
        }
    }

    fn impls() -> [impldoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(impltag(impldoc)) { some(impldoc) }
              _ { none }
            }
        }
    }

    fn types() -> [tydoc] {
        vec::filter_map(self) {|page|
            alt page {
              itempage(tytag(tydoc)) { some(tydoc) }
              _ { none }
            }
        }
    }
}

iface item {
    fn item() -> itemdoc;
}

impl of item for itemtag {
    fn item() -> itemdoc {
        alt self {
          doc::modtag(doc) { doc.item }
          doc::nmodtag(doc) { doc.item }
          doc::fntag(doc) { doc.item }
          doc::consttag(doc) { doc.item }
          doc::enumtag(doc) { doc.item }
          doc::restag(doc) { doc.item }
          doc::ifacetag(doc) { doc.item }
          doc::impltag(doc) { doc.item }
          doc::tytag(doc) { doc.item }
        }
    }
}

impl of item for simpleitemdoc {
    fn item() -> itemdoc { self.item }
}

impl of item for moddoc {
    fn item() -> itemdoc { self.item }
}

impl of item for nmoddoc {
    fn item() -> itemdoc { self.item }
}

impl of item for enumdoc {
    fn item() -> itemdoc { self.item }
}

impl of item for ifacedoc {
    fn item() -> itemdoc { self.item }
}

impl of item for impldoc {
    fn item() -> itemdoc { self.item }
}

impl util<A:item> for A {
    fn id() -> ast_id {
        self.item().id
    }

    fn name() -> str {
        self.item().name
    }

    fn path() -> [str] {
        self.item().path
    }

    fn brief() -> option<str> {
        self.item().brief
    }

    fn desc() -> option<str> {
        self.item().desc
    }

    fn sections() -> [section] {
        self.item().sections
    }
}