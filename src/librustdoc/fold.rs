// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use doc;
#[cfg(test)] use extract;
#[cfg(test)] use parse;

pub struct Fold<T> {
    ctxt: T,
    fold_doc: FoldDoc<T>,
    fold_crate: FoldCrate<T>,
    fold_item: FoldItem<T>,
    fold_mod: FoldMod<T>,
    fold_nmod: FoldNmod<T>,
    fold_fn: FoldFn<T>,
    fold_static: FoldStatic<T>,
    fold_enum: FoldEnum<T>,
    fold_trait: FoldTrait<T>,
    fold_impl: FoldImpl<T>,
    fold_type: FoldType<T>,
    fold_struct: FoldStruct<T>
}

impl<T:Clone> Clone for Fold<T> {
    fn clone(&self) -> Fold<T> {
        Fold {
            ctxt: self.ctxt.clone(),
            fold_doc: self.fold_doc,
            fold_crate: self.fold_crate,
            fold_item: self.fold_item,
            fold_mod: self.fold_mod,
            fold_nmod: self.fold_nmod,
            fold_fn: self.fold_fn,
            fold_static: self.fold_static,
            fold_enum: self.fold_enum,
            fold_trait: self.fold_trait,
            fold_impl: self.fold_impl,
            fold_type: self.fold_type,
            fold_struct: self.fold_struct
        }
    }
}

type FoldDoc<T> = @fn(fold: &Fold<T>, doc: doc::Doc) -> doc::Doc;
type FoldCrate<T> = @fn(fold: &Fold<T>, doc: doc::CrateDoc) -> doc::CrateDoc;
type FoldItem<T> = @fn(fold: &Fold<T>, doc: doc::ItemDoc) -> doc::ItemDoc;
type FoldMod<T> = @fn(fold: &Fold<T>, doc: doc::ModDoc) -> doc::ModDoc;
type FoldNmod<T> = @fn(fold: &Fold<T>, doc: doc::NmodDoc) -> doc::NmodDoc;
type FoldFn<T> = @fn(fold: &Fold<T>, doc: doc::FnDoc) -> doc::FnDoc;
type FoldStatic<T> = @fn(fold: &Fold<T>, doc: doc::StaticDoc) -> doc::StaticDoc;
type FoldEnum<T> = @fn(fold: &Fold<T>, doc: doc::EnumDoc) -> doc::EnumDoc;
type FoldTrait<T> = @fn(fold: &Fold<T>, doc: doc::TraitDoc) -> doc::TraitDoc;
type FoldImpl<T> = @fn(fold: &Fold<T>, doc: doc::ImplDoc) -> doc::ImplDoc;
type FoldType<T> = @fn(fold: &Fold<T>, doc: doc::TyDoc) -> doc::TyDoc;
type FoldStruct<T> = @fn(fold: &Fold<T>,
                         doc: doc::StructDoc) -> doc::StructDoc;

// This exists because fn types don't infer correctly as record
// initializers, but they do as function arguments
fn mk_fold<T>(
    ctxt: T,
    fold_doc: FoldDoc<T>,
    fold_crate: FoldCrate<T>,
    fold_item: FoldItem<T>,
    fold_mod: FoldMod<T>,
    fold_nmod: FoldNmod<T>,
    fold_fn: FoldFn<T>,
    fold_static: FoldStatic<T>,
    fold_enum: FoldEnum<T>,
    fold_trait: FoldTrait<T>,
    fold_impl: FoldImpl<T>,
    fold_type: FoldType<T>,
    fold_struct: FoldStruct<T>
) -> Fold<T> {
    Fold {
        ctxt: ctxt,
        fold_doc: fold_doc,
        fold_crate: fold_crate,
        fold_item: fold_item,
        fold_mod: fold_mod,
        fold_nmod: fold_nmod,
        fold_fn: fold_fn,
        fold_static: fold_static,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        fold_type: fold_type,
        fold_struct: fold_struct
    }
}

pub fn default_any_fold<T:Clone>(ctxt: T) -> Fold<T> {
    mk_fold(
        ctxt,
        |f, d| default_seq_fold_doc(f, d),
        |f, d| default_seq_fold_crate(f, d),
        |f, d| default_seq_fold_item(f, d),
        |f, d| default_any_fold_mod(f, d),
        |f, d| default_any_fold_nmod(f, d),
        |f, d| default_seq_fold_fn(f, d),
        |f, d| default_seq_fold_static(f, d),
        |f, d| default_seq_fold_enum(f, d),
        |f, d| default_seq_fold_trait(f, d),
        |f, d| default_seq_fold_impl(f, d),
        |f, d| default_seq_fold_type(f, d),
        |f, d| default_seq_fold_struct(f, d)
    )
}

pub fn default_seq_fold<T:Clone>(ctxt: T) -> Fold<T> {
    mk_fold(
        ctxt,
        |f, d| default_seq_fold_doc(f, d),
        |f, d| default_seq_fold_crate(f, d),
        |f, d| default_seq_fold_item(f, d),
        |f, d| default_seq_fold_mod(f, d),
        |f, d| default_seq_fold_nmod(f, d),
        |f, d| default_seq_fold_fn(f, d),
        |f, d| default_seq_fold_static(f, d),
        |f, d| default_seq_fold_enum(f, d),
        |f, d| default_seq_fold_trait(f, d),
        |f, d| default_seq_fold_impl(f, d),
        |f, d| default_seq_fold_type(f, d),
        |f, d| default_seq_fold_struct(f, d)
    )
}

pub fn default_par_fold<T:Clone>(ctxt: T) -> Fold<T> {
    mk_fold(
        ctxt,
        |f, d| default_seq_fold_doc(f, d),
        |f, d| default_seq_fold_crate(f, d),
        |f, d| default_seq_fold_item(f, d),
        |f, d| default_par_fold_mod(f, d),
        |f, d| default_par_fold_nmod(f, d),
        |f, d| default_seq_fold_fn(f, d),
        |f, d| default_seq_fold_static(f, d),
        |f, d| default_seq_fold_enum(f, d),
        |f, d| default_seq_fold_trait(f, d),
        |f, d| default_seq_fold_impl(f, d),
        |f, d| default_seq_fold_type(f, d),
        |f, d| default_seq_fold_struct(f, d)
    )
}

pub fn default_seq_fold_doc<T>(fold: &Fold<T>, doc: doc::Doc) -> doc::Doc {
    doc::Doc {
        pages: do doc.pages.iter().transform |page| {
            match (*page).clone() {
              doc::CratePage(doc) => {
                doc::CratePage((fold.fold_crate)(fold, doc))
              }
              doc::ItemPage(doc) => {
                doc::ItemPage(fold_ItemTag(fold, doc))
              }
            }
        }.collect(),
        .. doc
    }
}

pub fn default_seq_fold_crate<T>(
    fold: &Fold<T>,
    doc: doc::CrateDoc
) -> doc::CrateDoc {
    doc::CrateDoc {
        topmod: (fold.fold_mod)(fold, doc.topmod.clone())
    }
}

pub fn default_seq_fold_item<T>(
    _fold: &Fold<T>,
    doc: doc::ItemDoc
) -> doc::ItemDoc {
    doc
}

pub fn default_any_fold_mod<T:Clone>(
    fold: &Fold<T>,
    doc: doc::ModDoc
) -> doc::ModDoc {
    doc::ModDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        items: doc.items.iter().transform(|ItemTag| {
            fold_ItemTag(fold, (*ItemTag).clone())
        }).collect(),
        .. doc
    }
}

pub fn default_seq_fold_mod<T>(
    fold: &Fold<T>,
    doc: doc::ModDoc
) -> doc::ModDoc {
    doc::ModDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        items: doc.items.iter().transform(|ItemTag| {
            fold_ItemTag(fold, (*ItemTag).clone())
        }).collect(),
        .. doc
    }
}

pub fn default_par_fold_mod<T:Clone>(
    fold: &Fold<T>,
    doc: doc::ModDoc
) -> doc::ModDoc {
    doc::ModDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        items: doc.items.iter().transform(|ItemTag| {
            fold_ItemTag(fold, (*ItemTag).clone())
        }).collect(),
        .. doc
    }
}

pub fn default_any_fold_nmod<T:Clone>(
    fold: &Fold<T>,
    doc: doc::NmodDoc
) -> doc::NmodDoc {
    doc::NmodDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        fns: doc.fns.iter().transform(|FnDoc| {
            (fold.fold_fn)(fold, (*FnDoc).clone())
        }).collect(),
        .. doc
    }
}

pub fn default_seq_fold_nmod<T>(
    fold: &Fold<T>,
    doc: doc::NmodDoc
) -> doc::NmodDoc {
    doc::NmodDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        fns: doc.fns.iter().transform(|FnDoc| {
            (fold.fold_fn)(fold, (*FnDoc).clone())
        }).collect(),
        .. doc
    }
}

pub fn default_par_fold_nmod<T:Clone>(
    fold: &Fold<T>,
    doc: doc::NmodDoc
) -> doc::NmodDoc {
    doc::NmodDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        fns: doc.fns.iter().transform(|FnDoc| {
            (fold.fold_fn)(fold, (*FnDoc).clone())
        }).collect(),
        .. doc
    }
}

pub fn fold_ItemTag<T>(fold: &Fold<T>, doc: doc::ItemTag) -> doc::ItemTag {
    match doc {
      doc::ModTag(ModDoc) => {
        doc::ModTag((fold.fold_mod)(fold, ModDoc))
      }
      doc::NmodTag(nModDoc) => {
        doc::NmodTag((fold.fold_nmod)(fold, nModDoc))
      }
      doc::FnTag(FnDoc) => {
        doc::FnTag((fold.fold_fn)(fold, FnDoc))
      }
      doc::StaticTag(StaticDoc) => {
        doc::StaticTag((fold.fold_static)(fold, StaticDoc))
      }
      doc::EnumTag(EnumDoc) => {
        doc::EnumTag((fold.fold_enum)(fold, EnumDoc))
      }
      doc::TraitTag(TraitDoc) => {
        doc::TraitTag((fold.fold_trait)(fold, TraitDoc))
      }
      doc::ImplTag(ImplDoc) => {
        doc::ImplTag((fold.fold_impl)(fold, ImplDoc))
      }
      doc::TyTag(TyDoc) => {
        doc::TyTag((fold.fold_type)(fold, TyDoc))
      }
      doc::StructTag(StructDoc) => {
        doc::StructTag((fold.fold_struct)(fold, StructDoc))
      }
    }
}

pub fn default_seq_fold_fn<T>(
    fold: &Fold<T>,
    doc: doc::FnDoc
) -> doc::FnDoc {
    doc::SimpleItemDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

pub fn default_seq_fold_static<T>(
    fold: &Fold<T>,
    doc: doc::StaticDoc
) -> doc::StaticDoc {
    doc::SimpleItemDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

pub fn default_seq_fold_enum<T>(
    fold: &Fold<T>,
    doc: doc::EnumDoc
) -> doc::EnumDoc {
    doc::EnumDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

pub fn default_seq_fold_trait<T>(
    fold: &Fold<T>,
    doc: doc::TraitDoc
) -> doc::TraitDoc {
    doc::TraitDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

pub fn default_seq_fold_impl<T>(
    fold: &Fold<T>,
    doc: doc::ImplDoc
) -> doc::ImplDoc {
    doc::ImplDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

pub fn default_seq_fold_type<T>(
    fold: &Fold<T>,
    doc: doc::TyDoc
) -> doc::TyDoc {
    doc::SimpleItemDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

pub fn default_seq_fold_struct<T>(
    fold: &Fold<T>,
    doc: doc::StructDoc
) -> doc::StructDoc {
    doc::StructDoc {
        item: (fold.fold_item)(fold, doc.item.clone()),
        .. doc
    }
}

#[test]
fn default_fold_should_produce_same_doc() {
    let source = @"mod a { fn b() { } mod c { fn d() { } } }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_seq_fold(());
    let folded = (fld.fold_doc)(&fld, doc.clone());
    assert_eq!(doc, folded);
}

#[test]
fn default_fold_should_produce_same_statics() {
    let source = @"static a: int = 0;";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_seq_fold(());
    let folded = (fld.fold_doc)(&fld, doc.clone());
    assert_eq!(doc, folded);
}

#[test]
fn default_fold_should_produce_same_enums() {
    let source = @"enum a { b }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_seq_fold(());
    let folded = (fld.fold_doc)(&fld, doc.clone());
    assert_eq!(doc, folded);
}

#[test]
fn default_parallel_fold_should_produce_same_doc() {
    let source = @"mod a { fn b() { } mod c { fn d() { } } }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_par_fold(());
    let folded = (fld.fold_doc)(&fld, doc.clone());
    assert_eq!(doc, folded);
}
