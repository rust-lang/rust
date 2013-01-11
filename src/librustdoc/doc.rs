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

use core::prelude::*;

use doc;
use pass::Pass;

use core::cmp;
use core::option;
use core::vec;

pub type AstId = int;

pub type Doc_ = {
    pages: ~[Page]
};

impl Doc_ : cmp::Eq {
    pure fn eq(&self, other: &Doc_) -> bool {
        (*self).pages == (*other).pages
    }
    pure fn ne(&self, other: &Doc_) -> bool { !(*self).eq(other) }
}

pub enum Doc {
    Doc_(Doc_)
}

impl Doc : cmp::Eq {
    pure fn eq(&self, other: &Doc) -> bool { *(*self) == *(*other) }
    pure fn ne(&self, other: &Doc) -> bool { *(*self) != *(*other) }
}

pub enum Page {
    CratePage(CrateDoc),
    ItemPage(ItemTag)
}

impl Page : cmp::Eq {
    pure fn eq(&self, other: &Page) -> bool {
        match (*self) {
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
    pure fn ne(&self, other: &Page) -> bool { !(*self).eq(other) }
}

pub enum Implementation {
    Required,
    Provided,
}

impl Implementation : cmp::Eq {
    pure fn eq(&self, other: &Implementation) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    pure fn ne(&self, other: &Implementation) -> bool { !(*self).eq(other) }
}


/**
 * Most rustdocs can be parsed into 'sections' according to their markdown
 * headers
 */
pub type Section = {
    header: ~str,
    body: ~str
};

impl Section : cmp::Eq {
    pure fn eq(&self, other: &Section) -> bool {
        (*self).header == (*other).header && (*self).body == (*other).body
    }
    pure fn ne(&self, other: &Section) -> bool { !(*self).eq(other) }
}

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
pub type CrateDoc = {
    topmod: ModDoc,
};

impl CrateDoc : cmp::Eq {
    pure fn eq(&self, other: &CrateDoc) -> bool {
        (*self).topmod == (*other).topmod
    }
    pure fn ne(&self, other: &CrateDoc) -> bool { !(*self).eq(other) }
}

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

impl ItemTag : cmp::Eq {
    pure fn eq(&self, other: &ItemTag) -> bool {
        match (*self) {
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
    pure fn ne(&self, other: &ItemTag) -> bool { !(*self).eq(other) }
}

pub type ItemDoc = {
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
    pure fn eq(&self, other: &ItemDoc) -> bool {
        (*self).id == (*other).id &&
        (*self).name == (*other).name &&
        (*self).path == (*other).path &&
        (*self).brief == (*other).brief &&
        (*self).desc == (*other).desc &&
        (*self).sections == (*other).sections &&
        (*self).reexport == (*other).reexport
    }
    pure fn ne(&self, other: &ItemDoc) -> bool { !(*self).eq(other) }
}

pub type SimpleItemDoc = {
    item: ItemDoc,
    sig: Option<~str>
};

impl SimpleItemDoc : cmp::Eq {
    pure fn eq(&self, other: &SimpleItemDoc) -> bool {
        (*self).item == (*other).item && (*self).sig == (*other).sig
    }
    pure fn ne(&self, other: &SimpleItemDoc) -> bool { !(*self).eq(other) }
}

pub type ModDoc_ = {
    item: ItemDoc,
    items: ~[ItemTag],
    index: Option<Index>
};

impl ModDoc_ : cmp::Eq {
    pure fn eq(&self, other: &ModDoc_) -> bool {
        (*self).item == (*other).item &&
        (*self).items == (*other).items &&
        (*self).index == (*other).index
    }
    pure fn ne(&self, other: &ModDoc_) -> bool { !(*self).eq(other) }
}

pub enum ModDoc {
    ModDoc_(ModDoc_)
}

impl ModDoc : cmp::Eq {
    pure fn eq(&self, other: &ModDoc) -> bool { *(*self) == *(*other) }
    pure fn ne(&self, other: &ModDoc) -> bool { *(*self) != *(*other) }
}

pub type NmodDoc = {
    item: ItemDoc,
    fns: ~[FnDoc],
    index: Option<Index>
};

impl NmodDoc : cmp::Eq {
    pure fn eq(&self, other: &NmodDoc) -> bool {
        (*self).item == (*other).item &&
        (*self).fns == (*other).fns &&
        (*self).index == (*other).index
    }
    pure fn ne(&self, other: &NmodDoc) -> bool { !(*self).eq(other) }
}

pub type ConstDoc = SimpleItemDoc;

pub type FnDoc = SimpleItemDoc;

pub type EnumDoc = {
    item: ItemDoc,
    variants: ~[VariantDoc]
};

impl EnumDoc : cmp::Eq {
    pure fn eq(&self, other: &EnumDoc) -> bool {
        (*self).item == (*other).item && (*self).variants == (*other).variants
    }
    pure fn ne(&self, other: &EnumDoc) -> bool { !(*self).eq(other) }
}

pub type VariantDoc = {
    name: ~str,
    desc: Option<~str>,
    sig: Option<~str>
};

impl VariantDoc : cmp::Eq {
    pure fn eq(&self, other: &VariantDoc) -> bool {
        (*self).name == (*other).name &&
        (*self).desc == (*other).desc &&
        (*self).sig == (*other).sig
    }
    pure fn ne(&self, other: &VariantDoc) -> bool { !(*self).eq(other) }
}

pub type TraitDoc = {
    item: ItemDoc,
    methods: ~[MethodDoc]
};

impl TraitDoc : cmp::Eq {
    pure fn eq(&self, other: &TraitDoc) -> bool {
        (*self).item == (*other).item && (*self).methods == (*other).methods
    }
    pure fn ne(&self, other: &TraitDoc) -> bool { !(*self).eq(other) }
}

pub type MethodDoc = {
    name: ~str,
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    sig: Option<~str>,
    implementation: Implementation,
};

impl MethodDoc : cmp::Eq {
    pure fn eq(&self, other: &MethodDoc) -> bool {
        (*self).name == (*other).name &&
        (*self).brief == (*other).brief &&
        (*self).desc == (*other).desc &&
        (*self).sections == (*other).sections &&
        (*self).sig == (*other).sig &&
        (*self).implementation == (*other).implementation
    }
    pure fn ne(&self, other: &MethodDoc) -> bool { !(*self).eq(other) }
}

pub type ImplDoc = {
    item: ItemDoc,
    trait_types: ~[~str],
    self_ty: Option<~str>,
    methods: ~[MethodDoc]
};

impl ImplDoc : cmp::Eq {
    pure fn eq(&self, other: &ImplDoc) -> bool {
        (*self).item == (*other).item &&
        (*self).trait_types == (*other).trait_types &&
        (*self).self_ty == (*other).self_ty &&
        (*self).methods == (*other).methods
    }
    pure fn ne(&self, other: &ImplDoc) -> bool { !(*self).eq(other) }
}

pub type TyDoc = SimpleItemDoc;

pub type StructDoc = {
    item: ItemDoc,
    fields: ~[~str],
    sig: Option<~str>
};

impl StructDoc : cmp::Eq {
    pure fn eq(&self, other: &StructDoc) -> bool {
        return (*self).item == other.item
            && (*self).fields == other.fields
            && (*self).sig == other.sig;
    }
    pure fn ne(&self, other: &StructDoc) -> bool { !(*self).eq(other) }
}

pub type Index = {
    entries: ~[IndexEntry]
};

impl Index : cmp::Eq {
    pure fn eq(&self, other: &Index) -> bool {
        (*self).entries == (*other).entries
    }
    pure fn ne(&self, other: &Index) -> bool { !(*self).eq(other) }
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
pub type IndexEntry = {
    kind: ~str,
    name: ~str,
    brief: Option<~str>,
    link: ~str
};

impl IndexEntry : cmp::Eq {
    pure fn eq(&self, other: &IndexEntry) -> bool {
        (*self).kind == (*other).kind &&
        (*self).name == (*other).name &&
        (*self).brief == (*other).brief &&
        (*self).link == (*other).link
    }
    pure fn ne(&self, other: &IndexEntry) -> bool { !(*self).eq(other) }
}

impl Doc {
    fn CrateDoc() -> CrateDoc {
        option::get(vec::foldl(None, self.pages, |_m, page| {
            match *page {
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
            match *itemtag {
              ModTag(ModDoc) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods() -> ~[NmodDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              NmodTag(nModDoc) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns() -> ~[FnDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              FnTag(FnDoc) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts() -> ~[ConstDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              ConstTag(ConstDoc) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums() -> ~[EnumDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              EnumTag(EnumDoc) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits() -> ~[TraitDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              TraitTag(TraitDoc) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls() -> ~[ImplDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              ImplTag(ImplDoc) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types() -> ~[TyDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
              TyTag(TyDoc) => Some(TyDoc),
              _ => None
            }
        }
    }

    fn structs() -> ~[StructDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match *itemtag {
                StructTag(StructDoc) => Some(StructDoc),
                _ => None
            }
        }
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

impl ~[Page]: PageUtils {

    fn mods(&self) -> ~[ModDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(ModTag(ModDoc)) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods(&self) -> ~[NmodDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(NmodTag(nModDoc)) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns(&self) -> ~[FnDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(FnTag(FnDoc)) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts(&self) -> ~[ConstDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(ConstTag(ConstDoc)) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums(&self) -> ~[EnumDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(EnumTag(EnumDoc)) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits(&self) -> ~[TraitDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(TraitTag(TraitDoc)) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls(&self) -> ~[ImplDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(ImplTag(ImplDoc)) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types(&self) -> ~[TyDoc] {
        do vec::filter_map(*self) |page| {
            match *page {
              ItemPage(TyTag(TyDoc)) => Some(TyDoc),
              _ => None
            }
        }
    }
}

pub trait Item {
    pure fn item(&self) -> ItemDoc;
}

impl ItemTag: Item {
    pure fn item(&self) -> ItemDoc {
        match *self {
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
    pure fn item(&self) -> ItemDoc { self.item }
}

impl ModDoc: Item {
    pure fn item(&self) -> ItemDoc { self.item }
}

impl NmodDoc: Item {
    pure fn item(&self) -> ItemDoc { self.item }
}

impl EnumDoc: Item {
    pure fn item(&self) -> ItemDoc { self.item }
}

impl TraitDoc: Item {
    pure fn item(&self) -> ItemDoc { self.item }
}

impl ImplDoc: Item {
    pure fn item(&self) -> ItemDoc { self.item }
}

impl StructDoc: Item {
    pure fn item(&self) -> ItemDoc { self.item }
}

pub trait ItemUtils {
    pure fn id(&self) -> AstId;
    pure fn name(&self) -> ~str;
    pure fn path(&self) -> ~[~str];
    pure fn brief(&self) -> Option<~str>;
    pure fn desc(&self) -> Option<~str>;
    pure fn sections(&self) -> ~[Section];
}

impl<A:Item> A: ItemUtils {
    pure fn id(&self) -> AstId {
        self.item().id
    }

    pure fn name(&self) -> ~str {
        self.item().name
    }

    pure fn path(&self) -> ~[~str] {
        self.item().path
    }

    pure fn brief(&self) -> Option<~str> {
        self.item().brief
    }

    pure fn desc(&self) -> Option<~str> {
        self.item().desc
    }

    pure fn sections(&self) -> ~[Section] {
        self.item().sections
    }
}
