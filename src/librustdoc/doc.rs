// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The document model


use doc;

pub type AstId = int;

#[deriving(Clone, Eq)]
pub struct Doc {
    pages: ~[Page]
}

#[deriving(Clone, Eq)]
pub enum Page {
    CratePage(CrateDoc),
    ItemPage(ItemTag)
}

#[deriving(Clone, Eq)]
pub enum Implementation {
    Required,
    Provided,
}

/**
 * Most rustdocs can be parsed into 'sections' according to their markdown
 * headers
 */
#[deriving(Clone, Eq)]
pub struct Section {
    header: ~str,
    body: ~str
}

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
#[deriving(Clone, Eq)]
pub struct CrateDoc {
    topmod: ModDoc
}

#[deriving(Clone, Eq)]
pub enum ItemTag {
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

#[deriving(Clone, Eq)]
pub struct ItemDoc {
    id: AstId,
    name: ~str,
    path: ~[~str],
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    // Indicates that this node is a reexport of a different item
    reexport: bool
}

#[deriving(Clone, Eq)]
pub struct SimpleItemDoc {
    item: ItemDoc,
    sig: Option<~str>
}

#[deriving(Clone, Eq)]
pub struct ModDoc {
    item: ItemDoc,
    items: ~[ItemTag],
    index: Option<Index>
}

#[deriving(Clone, Eq)]
pub struct NmodDoc {
    item: ItemDoc,
    fns: ~[FnDoc],
    index: Option<Index>
}

pub type ConstDoc = SimpleItemDoc;

pub type FnDoc = SimpleItemDoc;

#[deriving(Clone, Eq)]
pub struct EnumDoc {
    item: ItemDoc,
    variants: ~[VariantDoc]
}

#[deriving(Clone, Eq)]
pub struct VariantDoc {
    name: ~str,
    desc: Option<~str>,
    sig: Option<~str>
}

#[deriving(Clone, Eq)]
pub struct TraitDoc {
    item: ItemDoc,
    methods: ~[MethodDoc]
}

#[deriving(Clone, Eq)]
pub struct MethodDoc {
    name: ~str,
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    sig: Option<~str>,
    implementation: Implementation,
}

#[deriving(Clone, Eq)]
pub struct ImplDoc {
    item: ItemDoc,
    bounds_str: Option<~str>,
    trait_types: ~[~str],
    self_ty: Option<~str>,
    methods: ~[MethodDoc]
}

pub type TyDoc = SimpleItemDoc;

#[deriving(Clone, Eq)]
pub struct StructDoc {
    item: ItemDoc,
    fields: ~[~str],
    sig: Option<~str>
}

#[deriving(Clone, Eq)]
pub struct Index {
    entries: ~[IndexEntry]
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
#[deriving(Clone, Eq)]
pub struct IndexEntry {
    kind: ~str,
    name: ~str,
    brief: Option<~str>,
    link: ~str
}

impl Doc {
    pub fn CrateDoc(&self) -> CrateDoc {
        self.pages.iter().fold(None, |_m, page| {
            match (*page).clone() {
              doc::CratePage(doc) => Some(doc),
              _ => None
            }
        }).unwrap()
    }

    pub fn cratemod(&self) -> ModDoc {
        self.CrateDoc().topmod.clone()
    }
}

macro_rules! filt_mapper {
    ($vec:expr, $pat:pat) => {
        do ($vec).iter().filter_map |thing| {
            match thing {
                &$pat => Some((*x).clone()),
                _ => None
            }
        }.collect()
    }
}

macro_rules! md {
    ($id:ident) => {
        filt_mapper!(self.items, $id(ref x))
    }
}
/// Some helper methods on ModDoc, mostly for testing
impl ModDoc {
    pub fn mods(&self) -> ~[ModDoc] {
        md!(ModTag)
    }

    pub fn nmods(&self) -> ~[NmodDoc] {
        md!(NmodTag)
    }

    pub fn fns(&self) -> ~[FnDoc] {
        md!(FnTag)
    }

    pub fn consts(&self) -> ~[ConstDoc] {
        md!(ConstTag)
    }

    pub fn enums(&self) -> ~[EnumDoc] {
        md!(EnumTag)
    }

    pub fn traits(&self) -> ~[TraitDoc] {
        md!(TraitTag)
    }

    pub fn impls(&self) -> ~[ImplDoc] {
        md!(ImplTag)
    }

    pub fn types(&self) -> ~[TyDoc] {
        md!(TyTag)
    }

    pub fn structs(&self) -> ~[StructDoc] {
        md!(StructTag)
    }
}

macro_rules! pu {
    ($id:ident) => {
        filt_mapper!(*self, ItemPage($id(ref x)))
    }
}

pub trait PageUtils {
    fn mods(&self) -> ~[ModDoc];
    fn nmods(&self) -> ~[NmodDoc];
    fn fns(&self) -> ~[FnDoc];
    fn consts(&self) -> ~[ConstDoc];
    fn enums(&self) -> ~[EnumDoc];
    fn traits(&self) -> ~[TraitDoc];
    fn impls(&self) -> ~[ImplDoc];
    fn types(&self) -> ~[TyDoc];
}

impl PageUtils for ~[Page] {

    fn mods(&self) -> ~[ModDoc] {
        pu!(ModTag)
    }

    fn nmods(&self) -> ~[NmodDoc] {
        pu!(NmodTag)
    }

    fn fns(&self) -> ~[FnDoc] {
        pu!(FnTag)
    }

    fn consts(&self) -> ~[ConstDoc] {
        pu!(ConstTag)
    }

    fn enums(&self) -> ~[EnumDoc] {
        pu!(EnumTag)
    }

    fn traits(&self) -> ~[TraitDoc] {
        pu!(TraitTag)
    }

    fn impls(&self) -> ~[ImplDoc] {
        pu!(ImplTag)
    }

    fn types(&self) -> ~[TyDoc] {
        pu!(TyTag)
    }
}

pub trait Item {
    fn item(&self) -> ItemDoc;
}

impl Item for ItemTag {
    fn item(&self) -> ItemDoc {
        match self {
          &doc::ModTag(ref doc) => doc.item.clone(),
          &doc::NmodTag(ref doc) => doc.item.clone(),
          &doc::FnTag(ref doc) => doc.item.clone(),
          &doc::ConstTag(ref doc) => doc.item.clone(),
          &doc::EnumTag(ref doc) => doc.item.clone(),
          &doc::TraitTag(ref doc) => doc.item.clone(),
          &doc::ImplTag(ref doc) => doc.item.clone(),
          &doc::TyTag(ref doc) => doc.item.clone(),
          &doc::StructTag(ref doc) => doc.item.clone(),
        }
    }
}

impl Item for SimpleItemDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

impl Item for ModDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

impl Item for NmodDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

impl Item for EnumDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

impl Item for TraitDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

impl Item for ImplDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

impl Item for StructDoc {
    fn item(&self) -> ItemDoc {
        self.item.clone()
    }
}

pub trait ItemUtils {
    fn id(&self) -> AstId;
    /// FIXME #5898: This conflicts with
    /// syntax::attr::AttrMetaMethods.name; This rustdoc seems to be on
    /// the way out so I'm making this one look bad rather than the
    /// new methods in attr.
    fn name_(&self) -> ~str;
    fn path(&self) -> ~[~str];
    fn brief(&self) -> Option<~str>;
    fn desc(&self) -> Option<~str>;
    fn sections(&self) -> ~[Section];
}

impl<A:Item> ItemUtils for A {
    fn id(&self) -> AstId {
        self.item().id
    }

    fn name_(&self) -> ~str {
        self.item().name.clone()
    }

    fn path(&self) -> ~[~str] {
        self.item().path.clone()
    }

    fn brief(&self) -> Option<~str> {
        self.item().brief.clone()
    }

    fn desc(&self) -> Option<~str> {
        self.item().desc.clone()
    }

    fn sections(&self) -> ~[Section] {
        self.item().sections.clone()
    }
}
