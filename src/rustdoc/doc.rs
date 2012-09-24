//! The document model

type AstId = int;

type Doc_ = {
    pages: ~[Page]
};

impl Doc_ : cmp::Eq {
    pure fn eq(other: &Doc_) -> bool {
        self.pages == (*other).pages
    }
    pure fn ne(other: &Doc_) -> bool { !self.eq(other) }
}

enum Doc {
    Doc_(Doc_)
}

impl Doc : cmp::Eq {
    pure fn eq(other: &Doc) -> bool { *self == *(*other) }
    pure fn ne(other: &Doc) -> bool { *self != *(*other) }
}

enum Page {
    CratePage(CrateDoc),
    ItemPage(ItemTag)
}

impl Page : cmp::Eq {
    pure fn eq(other: &Page) -> bool {
        match self {
            CratePage(e0a) => {
                match (*other) {
                    CratePage(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ItemPage(e0a) => {
                match (*other) {
                    ItemPage(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &Page) -> bool { !self.eq(other) }
}

enum Implementation {
    Required,
    Provided,
}

impl Implementation : cmp::Eq {
    pure fn eq(other: &Implementation) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &Implementation) -> bool { !self.eq(other) }
}


/**
 * Most rustdocs can be parsed into 'sections' according to their markdown
 * headers
 */
type Section = {
    header: ~str,
    body: ~str
};

impl Section : cmp::Eq {
    pure fn eq(other: &Section) -> bool {
        self.header == (*other).header && self.body == (*other).body
    }
    pure fn ne(other: &Section) -> bool { !self.eq(other) }
}

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
type CrateDoc = {
    topmod: ModDoc,
};

impl CrateDoc : cmp::Eq {
    pure fn eq(other: &CrateDoc) -> bool {
        self.topmod == (*other).topmod
    }
    pure fn ne(other: &CrateDoc) -> bool { !self.eq(other) }
}

enum ItemTag {
    ModTag(ModDoc),
    NmodTag(NmodDoc),
    ConstTag(ConstDoc),
    FnTag(FnDoc),
    EnumTag(EnumDoc),
    TraitTag(TraitDoc),
    ImplTag(ImplDoc),
    TyTag(TyDoc),
    StructTag(StructDoc)
}

impl ItemTag : cmp::Eq {
    pure fn eq(other: &ItemTag) -> bool {
        match self {
            ModTag(e0a) => {
                match (*other) {
                    ModTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            NmodTag(e0a) => {
                match (*other) {
                    NmodTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ConstTag(e0a) => {
                match (*other) {
                    ConstTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            FnTag(e0a) => {
                match (*other) {
                    FnTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            EnumTag(e0a) => {
                match (*other) {
                    EnumTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            TraitTag(e0a) => {
                match (*other) {
                    TraitTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            ImplTag(e0a) => {
                match (*other) {
                    ImplTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            TyTag(e0a) => {
                match (*other) {
                    TyTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
            StructTag(e0a) => {
                match (*other) {
                    StructTag(e0b) => e0a == e0b,
                    _ => false
                }
            }
        }
    }
    pure fn ne(other: &ItemTag) -> bool { !self.eq(other) }
}

type ItemDoc = {
    id: AstId,
    name: ~str,
    path: ~[~str],
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    // Indicates that this node is a reexport of a different item
    reexport: bool
};

impl ItemDoc : cmp::Eq {
    pure fn eq(other: &ItemDoc) -> bool {
        self.id == (*other).id &&
        self.name == (*other).name &&
        self.path == (*other).path &&
        self.brief == (*other).brief &&
        self.desc == (*other).desc &&
        self.sections == (*other).sections &&
        self.reexport == (*other).reexport
    }
    pure fn ne(other: &ItemDoc) -> bool { !self.eq(other) }
}

type SimpleItemDoc = {
    item: ItemDoc,
    sig: Option<~str>
};

impl SimpleItemDoc : cmp::Eq {
    pure fn eq(other: &SimpleItemDoc) -> bool {
        self.item == (*other).item && self.sig == (*other).sig
    }
    pure fn ne(other: &SimpleItemDoc) -> bool { !self.eq(other) }
}

type ModDoc_ = {
    item: ItemDoc,
    items: ~[ItemTag],
    index: Option<Index>
};

impl ModDoc_ : cmp::Eq {
    pure fn eq(other: &ModDoc_) -> bool {
        self.item == (*other).item &&
        self.items == (*other).items &&
        self.index == (*other).index
    }
    pure fn ne(other: &ModDoc_) -> bool { !self.eq(other) }
}

enum ModDoc {
    ModDoc_(ModDoc_)
}

impl ModDoc : cmp::Eq {
    pure fn eq(other: &ModDoc) -> bool { *self == *(*other) }
    pure fn ne(other: &ModDoc) -> bool { *self != *(*other) }
}

type NmodDoc = {
    item: ItemDoc,
    fns: ~[FnDoc],
    index: Option<Index>
};

impl NmodDoc : cmp::Eq {
    pure fn eq(other: &NmodDoc) -> bool {
        self.item == (*other).item &&
        self.fns == (*other).fns &&
        self.index == (*other).index
    }
    pure fn ne(other: &NmodDoc) -> bool { !self.eq(other) }
}

type ConstDoc = SimpleItemDoc;

type FnDoc = SimpleItemDoc;

type EnumDoc = {
    item: ItemDoc,
    variants: ~[VariantDoc]
};

impl EnumDoc : cmp::Eq {
    pure fn eq(other: &EnumDoc) -> bool {
        self.item == (*other).item && self.variants == (*other).variants
    }
    pure fn ne(other: &EnumDoc) -> bool { !self.eq(other) }
}

type VariantDoc = {
    name: ~str,
    desc: Option<~str>,
    sig: Option<~str>
};

impl VariantDoc : cmp::Eq {
    pure fn eq(other: &VariantDoc) -> bool {
        self.name == (*other).name &&
        self.desc == (*other).desc &&
        self.sig == (*other).sig
    }
    pure fn ne(other: &VariantDoc) -> bool { !self.eq(other) }
}

type TraitDoc = {
    item: ItemDoc,
    methods: ~[MethodDoc]
};

impl TraitDoc : cmp::Eq {
    pure fn eq(other: &TraitDoc) -> bool {
        self.item == (*other).item && self.methods == (*other).methods
    }
    pure fn ne(other: &TraitDoc) -> bool { !self.eq(other) }
}

type MethodDoc = {
    name: ~str,
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    sig: Option<~str>,
    implementation: Implementation,
};

impl MethodDoc : cmp::Eq {
    pure fn eq(other: &MethodDoc) -> bool {
        self.name == (*other).name &&
        self.brief == (*other).brief &&
        self.desc == (*other).desc &&
        self.sections == (*other).sections &&
        self.sig == (*other).sig &&
        self.implementation == (*other).implementation
    }
    pure fn ne(other: &MethodDoc) -> bool { !self.eq(other) }
}

type ImplDoc = {
    item: ItemDoc,
    trait_types: ~[~str],
    self_ty: Option<~str>,
    methods: ~[MethodDoc]
};

impl ImplDoc : cmp::Eq {
    pure fn eq(other: &ImplDoc) -> bool {
        self.item == (*other).item &&
        self.trait_types == (*other).trait_types &&
        self.self_ty == (*other).self_ty &&
        self.methods == (*other).methods
    }
    pure fn ne(other: &ImplDoc) -> bool { !self.eq(other) }
}

type TyDoc = SimpleItemDoc;

type StructDoc = {
    item: ItemDoc,
    fields: ~[~str],
    sig: Option<~str>
};

impl StructDoc : cmp::Eq {
    pure fn eq(other: &StructDoc) -> bool {
        return self.item == other.item
            && self.fields == other.fields
            && self.sig == other.sig;
    }
    pure fn ne(other: &StructDoc) -> bool { !self.eq(other) }
}

type Index = {
    entries: ~[IndexEntry]
};

impl Index : cmp::Eq {
    pure fn eq(other: &Index) -> bool {
        self.entries == (*other).entries
    }
    pure fn ne(other: &Index) -> bool { !self.eq(other) }
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
type IndexEntry = {
    kind: ~str,
    name: ~str,
    brief: Option<~str>,
    link: ~str
};

impl IndexEntry : cmp::Eq {
    pure fn eq(other: &IndexEntry) -> bool {
        self.kind == (*other).kind &&
        self.name == (*other).name &&
        self.brief == (*other).brief &&
        self.link == (*other).link
    }
    pure fn ne(other: &IndexEntry) -> bool { !self.eq(other) }
}

impl Doc {
    fn CrateDoc() -> CrateDoc {
        option::get(&vec::foldl(None, self.pages, |_m, page| {
            match page {
              doc::CratePage(doc) => Some(doc),
              _ => None
            }
        }))
    }

    fn cratemod() -> ModDoc {
        self.CrateDoc().topmod
    }
}

/// Some helper methods on ModDoc, mostly for testing
impl ModDoc {

    fn mods() -> ~[ModDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              ModTag(ModDoc) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods() -> ~[NmodDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              NmodTag(nModDoc) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns() -> ~[FnDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              FnTag(FnDoc) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts() -> ~[ConstDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              ConstTag(ConstDoc) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums() -> ~[EnumDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              EnumTag(EnumDoc) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits() -> ~[TraitDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              TraitTag(TraitDoc) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls() -> ~[ImplDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              ImplTag(ImplDoc) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types() -> ~[TyDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
              TyTag(TyDoc) => Some(TyDoc),
              _ => None
            }
        }
    }

    fn structs() -> ~[StructDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match itemtag {
                StructTag(StructDoc) => Some(StructDoc),
                _ => None
            }
        }
    }
}

trait PageUtils {
    fn mods() -> ~[ModDoc];
    fn nmods() -> ~[NmodDoc];
    fn fns() -> ~[FnDoc];
    fn consts() -> ~[ConstDoc];
    fn enums() -> ~[EnumDoc];
    fn traits() -> ~[TraitDoc];
    fn impls() -> ~[ImplDoc];
    fn types() -> ~[TyDoc];
}

impl ~[Page]: PageUtils {

    fn mods() -> ~[ModDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(ModTag(ModDoc)) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods() -> ~[NmodDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(NmodTag(nModDoc)) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns() -> ~[FnDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(FnTag(FnDoc)) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts() -> ~[ConstDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(ConstTag(ConstDoc)) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums() -> ~[EnumDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(EnumTag(EnumDoc)) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits() -> ~[TraitDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(TraitTag(TraitDoc)) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls() -> ~[ImplDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(ImplTag(ImplDoc)) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types() -> ~[TyDoc] {
        do vec::filter_map(self) |page| {
            match page {
              ItemPage(TyTag(TyDoc)) => Some(TyDoc),
              _ => None
            }
        }
    }
}

trait Item {
    pure fn item() -> ItemDoc;
}

impl ItemTag: Item {
    pure fn item() -> ItemDoc {
        match self {
          doc::ModTag(doc) => doc.item,
          doc::NmodTag(doc) => doc.item,
          doc::FnTag(doc) => doc.item,
          doc::ConstTag(doc) => doc.item,
          doc::EnumTag(doc) => doc.item,
          doc::TraitTag(doc) => doc.item,
          doc::ImplTag(doc) => doc.item,
          doc::TyTag(doc) => doc.item,
          doc::StructTag(doc) => doc.item
        }
    }
}

impl SimpleItemDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

impl ModDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

impl NmodDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

impl EnumDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

impl TraitDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

impl ImplDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

impl StructDoc: Item {
    pure fn item() -> ItemDoc { self.item }
}

trait ItemUtils {
    pure fn id() -> AstId;
    pure fn name() -> ~str;
    pure fn path() -> ~[~str];
    pure fn brief() -> Option<~str>;
    pure fn desc() -> Option<~str>;
    pure fn sections() -> ~[Section];
}

impl<A:Item> A: ItemUtils {
    pure fn id() -> AstId {
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

    pure fn sections() -> ~[Section] {
        self.item().sections
    }
}
