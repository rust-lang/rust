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

#[deriving_eq]
pub struct Doc {
    pages: ~[Page]
}

#[deriving_eq]
pub enum Page {
    CratePage(CrateDoc),
    ItemPage(ItemTag)
}

#[deriving_eq]
pub enum Implementation {
    Required,
    Provided,
}

/**
 * Most rustdocs can be parsed into 'sections' according to their markdown
 * headers
 */
#[deriving_eq]
pub struct Section {
    header: ~str,
    body: ~str
}

// FIXME (#2596): We currently give topmod the name of the crate.  There
// would probably be fewer special cases if the crate had its own name
// and topmod's name was the empty string.
#[deriving_eq]
pub struct CrateDoc {
    topmod: ModDoc
}

#[deriving_eq]
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

#[deriving_eq]
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

#[deriving_eq]
pub struct SimpleItemDoc {
    item: ItemDoc,
    sig: Option<~str>
}

#[deriving_eq]
pub struct ModDoc {
    item: ItemDoc,
    items: ~[ItemTag],
    index: Option<Index>
}

#[deriving_eq]
pub struct NmodDoc {
    item: ItemDoc,
    fns: ~[FnDoc],
    index: Option<Index>
}

pub type ConstDoc = SimpleItemDoc;

pub type FnDoc = SimpleItemDoc;

#[deriving_eq]
pub struct EnumDoc {
    item: ItemDoc,
    variants: ~[VariantDoc]
}

#[deriving_eq]
pub struct VariantDoc {
    name: ~str,
    desc: Option<~str>,
    sig: Option<~str>
}

#[deriving_eq]
pub struct TraitDoc {
    item: ItemDoc,
    methods: ~[MethodDoc]
}

#[deriving_eq]
pub struct MethodDoc {
    name: ~str,
    brief: Option<~str>,
    desc: Option<~str>,
    sections: ~[Section],
    sig: Option<~str>,
    implementation: Implementation,
}

#[deriving_eq]
pub struct ImplDoc {
    item: ItemDoc,
    trait_types: ~[~str],
    self_ty: Option<~str>,
    methods: ~[MethodDoc]
}

pub type TyDoc = SimpleItemDoc;

#[deriving_eq]
pub struct StructDoc {
    item: ItemDoc,
    fields: ~[~str],
    sig: Option<~str>
}

#[deriving_eq]
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
#[deriving_eq]
pub struct IndexEntry {
    kind: ~str,
    name: ~str,
    brief: Option<~str>,
    link: ~str
}

impl Doc {
    fn CrateDoc() -> CrateDoc {
        option::get(vec::foldl(None, self.pages, |_m, page| {
            match copy *page {
              doc::CratePage(doc) => Some(doc),
              _ => None
            }
        }))
    }

    fn cratemod() -> ModDoc {
        copy self.CrateDoc().topmod
    }
}

/// Some helper methods on ModDoc, mostly for testing
impl ModDoc {
    fn mods() -> ~[ModDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              ModTag(ModDoc) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods() -> ~[NmodDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              NmodTag(nModDoc) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns() -> ~[FnDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              FnTag(FnDoc) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts() -> ~[ConstDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              ConstTag(ConstDoc) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums() -> ~[EnumDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              EnumTag(EnumDoc) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits() -> ~[TraitDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              TraitTag(TraitDoc) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls() -> ~[ImplDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              ImplTag(ImplDoc) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types() -> ~[TyDoc] {
        do vec::filter_map(self.items) |itemtag| {
            match copy *itemtag {
              TyTag(TyDoc) => Some(TyDoc),
              _ => None
            }
        }
    }

    fn structs() -> ~[StructDoc] {
        do vec::filter_map(self.items) |itemtag| {
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

impl ~[Page]: PageUtils {

    fn mods(&self) -> ~[ModDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(ModTag(ModDoc)) => Some(ModDoc),
              _ => None
            }
        }
    }

    fn nmods(&self) -> ~[NmodDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(NmodTag(nModDoc)) => Some(nModDoc),
              _ => None
            }
        }
    }

    fn fns(&self) -> ~[FnDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(FnTag(FnDoc)) => Some(FnDoc),
              _ => None
            }
        }
    }

    fn consts(&self) -> ~[ConstDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(ConstTag(ConstDoc)) => Some(ConstDoc),
              _ => None
            }
        }
    }

    fn enums(&self) -> ~[EnumDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(EnumTag(EnumDoc)) => Some(EnumDoc),
              _ => None
            }
        }
    }

    fn traits(&self) -> ~[TraitDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(TraitTag(TraitDoc)) => Some(TraitDoc),
              _ => None
            }
        }
    }

    fn impls(&self) -> ~[ImplDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
              ItemPage(ImplTag(ImplDoc)) => Some(ImplDoc),
              _ => None
            }
        }
    }

    fn types(&self) -> ~[TyDoc] {
        do vec::filter_map(*self) |page| {
            match copy *page {
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

impl SimpleItemDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
}

impl ModDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
}

impl NmodDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
}

impl EnumDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
}

impl TraitDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
}

impl ImplDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
}

impl StructDoc: Item {
    pure fn item(&self) -> ItemDoc { copy self.item }
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
        copy self.item().name
    }

    pure fn path(&self) -> ~[~str] {
        copy self.item().path
    }

    pure fn brief(&self) -> Option<~str> {
        copy self.item().brief
    }

    pure fn desc(&self) -> Option<~str> {
        copy self.item().desc
    }

    pure fn sections(&self) -> ~[Section] {
        copy self.item().sections
    }
}
