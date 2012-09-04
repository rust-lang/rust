//! The document model

type ast_id = int;

type doc_ = {
    pages: ~[page]
};

impl doc_ : cmp::Eq {
    pure fn eq(&&other: doc_) -> bool {
        self.pages == other.pages
    }
}

enum doc {
    doc_(doc_)
}

impl doc : cmp::Eq {
    pure fn eq(&&other: doc) -> bool {
        *self == *other
    }
}

enum page {
    cratepage(cratedoc),
    itempage(itemtag)
}

impl page : cmp::Eq {
    pure fn eq(&&other: page) -> bool {
        match self {
            cratepage(e0a) => {
                match other {
                    cratepage(e0b) => e0a == e0b,
                    _ => false
                }
            }
            itempage(e0a) => {
                match other {
                    itempage(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
}

enum implementation {
    required,
    provided,
}

impl implementation : cmp::Eq {
    pure fn eq(&&other: implementation) -> bool {
        (self as uint) == (other as uint)
    }
}


/**
 * Most rustdocs can be parsed into 'sections' according to their markdown
 * headers
 */
type section = {
    header: ~str,
    body: ~str
};

impl section : cmp::Eq {
    pure fn eq(&&other: section) -> bool {
        self.header == other.header && self.body == other.body
    }
}

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
type cratedoc = {
    topmod: moddoc,
};

impl cratedoc : cmp::Eq {
    pure fn eq(&&other: cratedoc) -> bool {
        self.topmod == other.topmod
    }
}

enum itemtag {
    modtag(moddoc),
    nmodtag(nmoddoc),
    consttag(constdoc),
    fntag(fndoc),
    enumtag(enumdoc),
    traittag(traitdoc),
    impltag(impldoc),
    tytag(tydoc)
}

impl itemtag : cmp::Eq {
    pure fn eq(&&other: itemtag) -> bool {
        match self {
            modtag(e0a) => {
                match other {
                    modtag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            nmodtag(e0a) => {
                match other {
                    nmodtag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            consttag(e0a) => {
                match other {
                    consttag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            fntag(e0a) => {
                match other {
                    fntag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            enumtag(e0a) => {
                match other {
                    enumtag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            traittag(e0a) => {
                match other {
                    traittag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            impltag(e0a) => {
                match other {
                    impltag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            tytag(e0a) => {
                match other {
                    tytag(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
}

type itemdoc = {
    id: ast_id,
    name: ~str,
    path: ~[~str],
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[section],
    // Indicates that this node is a reexport of a different item
    reexport: bool
};

impl itemdoc : cmp::Eq {
    pure fn eq(&&other: itemdoc) -> bool {
        self.id == other.id &&
        self.name == other.name &&
        self.path == other.path &&
        self.brief == other.brief &&
        self.desc == other.desc &&
        self.sections == other.sections &&
        self.reexport == other.reexport
    }
}

type simpleitemdoc = {
    item: itemdoc,
    sig: Option<~str>
};

impl simpleitemdoc : cmp::Eq {
    pure fn eq(&&other: simpleitemdoc) -> bool {
        self.item == other.item && self.sig == other.sig
    }
}

type moddoc_ = {
    item: itemdoc,
    items: ~[itemtag],
    index: Option<index>
};

impl moddoc_ : cmp::Eq {
    pure fn eq(&&other: moddoc_) -> bool {
        self.item == other.item &&
        self.items == other.items &&
        self.index == other.index
    }
}

enum moddoc {
    moddoc_(moddoc_)
}

impl moddoc : cmp::Eq {
    pure fn eq(&&other: moddoc) -> bool {
        *self == *other
    }
}

type nmoddoc = {
    item: itemdoc,
    fns: ~[fndoc],
    index: Option<index>
};

impl nmoddoc : cmp::Eq {
    pure fn eq(&&other: nmoddoc) -> bool {
        self.item == other.item &&
        self.fns == other.fns &&
        self.index == other.index
    }
}

type constdoc = simpleitemdoc;

type fndoc = simpleitemdoc;

type enumdoc = {
    item: itemdoc,
    variants: ~[variantdoc]
};

impl enumdoc : cmp::Eq {
    pure fn eq(&&other: enumdoc) -> bool {
        self.item == other.item && self.variants == other.variants
    }
}

type variantdoc = {
    name: ~str,
    desc: Option<~str>,
    sig: Option<~str>
};

impl variantdoc : cmp::Eq {
    pure fn eq(&&other: variantdoc) -> bool {
        self.name == other.name &&
        self.desc == other.desc &&
        self.sig == other.sig
    }
}

type traitdoc = {
    item: itemdoc,
    methods: ~[methoddoc]
};

impl traitdoc : cmp::Eq {
    pure fn eq(&&other: traitdoc) -> bool {
        self.item == other.item && self.methods == other.methods
    }
}

type methoddoc = {
    name: ~str,
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[section],
    sig: Option<~str>,
    implementation: implementation,
};

impl methoddoc : cmp::Eq {
    pure fn eq(&&other: methoddoc) -> bool {
        self.name == other.name &&
        self.brief == other.brief &&
        self.desc == other.desc &&
        self.sections == other.sections &&
        self.sig == other.sig &&
        self.implementation == other.implementation
    }
}

type impldoc = {
    item: itemdoc,
    trait_types: ~[~str],
    self_ty: Option<~str>,
    methods: ~[methoddoc]
};

impl impldoc : cmp::Eq {
    pure fn eq(&&other: impldoc) -> bool {
        self.item == other.item &&
        self.trait_types == other.trait_types &&
        self.self_ty == other.self_ty &&
        self.methods == other.methods
    }
}

type tydoc = simpleitemdoc;

type index = {
    entries: ~[index_entry]
};

impl index : cmp::Eq {
    pure fn eq(&&other: index) -> bool {
        self.entries == other.entries
    }
}

/**
 * A single entry in an index
 *
 * Fields:
 *
 * * kind - The type of thing being indexed, e.g. 'Module'
 * * name - The name of the thing
 * * brief - The brief description
 * * link - A format-specific string representing the link target
 */
type index_entry = {
    kind: ~str,
    name: ~str,
    brief: Option<~str>,
    link: ~str
};

impl index_entry : cmp::Eq {
    pure fn eq(&&other: index_entry) -> bool {
        self.kind == other.kind &&
        self.name == other.name &&
        self.brief == other.brief &&
        self.link == other.link
    }
}

impl doc {
    fn cratedoc() -> cratedoc {
        option::get(vec::foldl(None, self.pages, |_m, page| {
            match page {
              doc::cratepage(doc) => Some(doc),
              _ => None
            }
        }))
    }

    fn cratemod() -> moddoc {
        self.cratedoc().topmod
    }
}

/// Some helper methods on moddoc, mostly for testing
impl moddoc {

    fn mods() -> ~[moddoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              modtag(moddoc) => Some(moddoc),
              _ => None
            }
        }
    }

    fn nmods() -> ~[nmoddoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              nmodtag(nmoddoc) => Some(nmoddoc),
              _ => None
            }
        }
    }

    fn fns() -> ~[fndoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              fntag(fndoc) => Some(fndoc),
              _ => None
            }
        }
    }

    fn consts() -> ~[constdoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              consttag(constdoc) => Some(constdoc),
              _ => None
            }
        }
    }

    fn enums() -> ~[enumdoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              enumtag(enumdoc) => Some(enumdoc),
              _ => None
            }
        }
    }

    fn traits() -> ~[traitdoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              traittag(traitdoc) => Some(traitdoc),
              _ => None
            }
        }
    }

    fn impls() -> ~[impldoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              impltag(impldoc) => Some(impldoc),
              _ => None
            }
        }
    }

    fn types() -> ~[tydoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              tytag(tydoc) => Some(tydoc),
              _ => None
            }
        }
    }
}

trait page_utils {
    fn mods() -> ~[moddoc];
    fn nmods() -> ~[nmoddoc];
    fn fns() -> ~[fndoc];
    fn consts() -> ~[constdoc];
    fn enums() -> ~[enumdoc];
    fn traits() -> ~[traitdoc];
    fn impls() -> ~[impldoc];
    fn types() -> ~[tydoc];
}

impl ~[page]: page_utils {

    fn mods() -> ~[moddoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(modtag(moddoc)) => Some(moddoc),
              _ => None
            }
        }
    }

    fn nmods() -> ~[nmoddoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(nmodtag(nmoddoc)) => Some(nmoddoc),
              _ => None
            }
        }
    }

    fn fns() -> ~[fndoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(fntag(fndoc)) => Some(fndoc),
              _ => None
            }
        }
    }

    fn consts() -> ~[constdoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(consttag(constdoc)) => Some(constdoc),
              _ => None
            }
        }
    }

    fn enums() -> ~[enumdoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(enumtag(enumdoc)) => Some(enumdoc),
              _ => None
            }
        }
    }

    fn traits() -> ~[traitdoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(traittag(traitdoc)) => Some(traitdoc),
              _ => None
            }
        }
    }

    fn impls() -> ~[impldoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(impltag(impldoc)) => Some(impldoc),
              _ => None
            }
        }
    }

    fn types() -> ~[tydoc] {
        do vec::filter_map(self) |page| {
            match page {
              itempage(tytag(tydoc)) => Some(tydoc),
              _ => None
            }
        }
    }
}

trait item {
    pure fn item() -> itemdoc;
}

impl itemtag: item {
    pure fn item() -> itemdoc {
        match self {
          doc::modtag(doc) => doc.item,
          doc::nmodtag(doc) => doc.item,
          doc::fntag(doc) => doc.item,
          doc::consttag(doc) => doc.item,
          doc::enumtag(doc) => doc.item,
          doc::traittag(doc) => doc.item,
          doc::impltag(doc) => doc.item,
          doc::tytag(doc) => doc.item
        }
    }
}

impl simpleitemdoc: item {
    pure fn item() -> itemdoc { self.item }
}

impl moddoc: item {
    pure fn item() -> itemdoc { self.item }
}

impl nmoddoc: item {
    pure fn item() -> itemdoc { self.item }
}

impl enumdoc: item {
    pure fn item() -> itemdoc { self.item }
}

impl traitdoc: item {
    pure fn item() -> itemdoc { self.item }
}

impl impldoc: item {
    pure fn item() -> itemdoc { self.item }
}

trait item_utils {
    pure fn id() -> ast_id;
    pure fn name() -> ~str;
    pure fn path() -> ~[~str];
    pure fn brief() -> Option<~str>;
    pure fn desc() -> Option<~str>;
    pure fn sections() -> ~[section];
}

impl<A:item> A: item_utils {
    pure fn id() -> ast_id {
        self.item().id
    }

    pure fn name() -> ~str {
        self.item().name
    }

    pure fn path() -> ~[~str] {
        self.item().path
    }

    pure fn brief() -> Option<~str> {
        self.item().brief
    }

    pure fn desc() -> Option<~str> {
        self.item().desc
    }

    pure fn sections() -> ~[section] {
        self.item().sections
    }
}
