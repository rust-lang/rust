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

#[deriving(Eq)]
pub struct Doc {
    pages: ~[Page]
}

#[deriving(Eq)]
pub enum Page {
    CratePage(CrateDoc),
    ItemPage(ItemTag)
}

#[deriving(Eq)]
pub enum Implementation {
    Required,
    Provided,
}

/**
 * Most rustdocs can be parsed into 'sections' according to their markdown
 * headers
 */
#[deriving(Eq)]
pub struct Section {
    header: ~str,
    body: ~str
}

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
#[deriving(Eq)]
pub struct CrateDoc {
    topmod: ModDoc
}

#[deriving(Eq)]
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

#[deriving(Eq)]
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

#[deriving(Eq)]
pub struct SimpleItemDoc {
    item: ItemDoc,
    sig: Option<~str>
}

#[deriving(Eq)]
pub struct ModDoc {
    item: ItemDoc,
    items: ~[ItemTag],
    index: Option<Index>
}

#[deriving(Eq)]
pub struct NmodDoc {
    item: ItemDoc,
    fns: ~[FnDoc],
    index: Option<Index>
}

pub type ConstDoc = SimpleItemDoc;

pub type FnDoc = SimpleItemDoc;

#[deriving(Eq)]
pub struct EnumDoc {
    item: ItemDoc,
    variants: ~[VariantDoc]
}

#[deriving(Eq)]
pub struct VariantDoc {
    name: ~str,
    desc: Option<~str>,
    sig: Option<~str>
}

#[deriving(Eq)]
pub struct TraitDoc {
    item: ItemDoc,
    methods: ~[MethodDoc]
}

#[deriving(Eq)]
pub struct MethodDoc {
    name: ~str,
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    sig: Option<~str>,
    implementation: Implementation,
}

#[deriving(Eq)]
pub struct ImplDoc {
    item: ItemDoc,
    bounds_str: Option<~str>,
    trait_types: ~[~str],
    self_ty: Option<~str>,
    methods: ~[MethodDoc]
}

pub type TyDoc = SimpleItemDoc;

#[deriving(Eq)]
pub struct StructDoc {
    item: ItemDoc,
    fields: ~[~str],
    sig: Option<~str>
}

#[deriving(Eq)]
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
#[deriving(Eq)]
pub struct IndexEntry {
    kind: ~str,
    name: ~str,
    brief: Option<~str>,
    link: ~str
}

pub impl Doc {
    fn CrateDoc(&self) -> CrateDoc {
        vec::foldl(None, self.pages, |_m, page| {
            match copy *page {
              doc::CratePage(doc) => Some(doc),
              _ => None
            }
        }).get()
    }

    fn cratemod(&self) -> ModDoc {
        copy self.CrateDoc().topmod
    }
}

/// Some helper methods on ModDoc, mostly for testing
pub impl ModDoc {
    fn mods(&self) -> ~[ModDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              ModTag(ModDoc) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods(&self) -> ~[NmodDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              NmodTag(nModDoc) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns(&self) -> ~[FnDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              FnTag(FnDoc) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts(&self) -> ~[ConstDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              ConstTag(ConstDoc) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums(&self) -> ~[EnumDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              EnumTag(EnumDoc) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits(&self) -> ~[TraitDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              TraitTag(TraitDoc) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls(&self) -> ~[ImplDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              ImplTag(ImplDoc) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types(&self) -> ~[TyDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
              TyTag(TyDoc) => Some(TyDoc),
              _ => None
            }
        }
    }

    fn structs(&self) -> ~[StructDoc] {
        do vec::filter_mapped(self.items) |itemtag| {
            match copy *itemtag {
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

impl PageUtils for ~[Page] {

    fn mods(&self) -> ~[ModDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(ModTag(ModDoc)) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods(&self) -> ~[NmodDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(NmodTag(nModDoc)) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns(&self) -> ~[FnDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(FnTag(FnDoc)) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts(&self) -> ~[ConstDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(ConstTag(ConstDoc)) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums(&self) -> ~[EnumDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(EnumTag(EnumDoc)) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits(&self) -> ~[TraitDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(TraitTag(TraitDoc)) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls(&self) -> ~[ImplDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(ImplTag(ImplDoc)) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types(&self) -> ~[TyDoc] {
        do vec::filter_mapped(*self) |page| {
            match copy *page {
              ItemPage(TyTag(TyDoc)) => Some(TyDoc),
              _ => None
            }
        }
    }
}

pub trait Item {
    fn item(&self) -> ItemDoc;
}

impl Item for ItemTag {
    fn item(&self) -> ItemDoc {
        match self {
          &doc::ModTag(ref doc) => copy doc.item,
          &doc::NmodTag(ref doc) => copy doc.item,
          &doc::FnTag(ref doc) => copy doc.item,
          &doc::ConstTag(ref doc) => copy doc.item,
          &doc::EnumTag(ref doc) => copy doc.item,
          &doc::TraitTag(ref doc) => copy doc.item,
          &doc::ImplTag(ref doc) => copy doc.item,
          &doc::TyTag(ref doc) => copy doc.item,
          &doc::StructTag(ref doc) => copy doc.item
        }
    }
}

impl Item for SimpleItemDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

impl Item for ModDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

impl Item for NmodDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

impl Item for EnumDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

impl Item for TraitDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

impl Item for ImplDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

impl Item for StructDoc {
    fn item(&self) -> ItemDoc { copy self.item }
}

pub trait ItemUtils {
    fn id(&self) -> AstId;
    fn name(&self) -> ~str;
    fn path(&self) -> ~[~str];
    fn brief(&self) -> Option<~str>;
    fn desc(&self) -> Option<~str>;
    fn sections(&self) -> ~[Section];
}

impl<A:Item> ItemUtils for A {
    fn id(&self) -> AstId {
        self.item().id
    }

    fn name(&self) -> ~str {
        copy self.item().name
    }

    fn path(&self) -> ~[~str] {
        copy self.item().path
    }

    fn brief(&self) -> Option<~str> {
        copy self.item().brief
    }

    fn desc(&self) -> Option<~str> {
        copy self.item().desc
    }

    fn sections(&self) -> ~[Section] {
        copy self.item().sections
    }
}
