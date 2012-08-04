export fold;
export default_seq_fold;
export default_seq_fold_doc;
export default_seq_fold_crate;
export default_seq_fold_item;
export default_seq_fold_mod;
export default_seq_fold_nmod;
export default_seq_fold_fn;
export default_seq_fold_const;
export default_seq_fold_enum;
export default_seq_fold_trait;
export default_seq_fold_impl;
export default_seq_fold_type;
export default_par_fold;
export default_par_fold_mod;
export default_par_fold_nmod;
export default_any_fold;
export default_any_fold_mod;
export default_any_fold_nmod;

enum fold<T> = t<T>;

type fold_doc<T> = fn~(fold: fold<T>, doc: doc::doc) -> doc::doc;
type fold_crate<T> = fn~(fold: fold<T>, doc: doc::cratedoc) -> doc::cratedoc;
type fold_item<T> = fn~(fold: fold<T>, doc: doc::itemdoc) -> doc::itemdoc;
type fold_mod<T> = fn~(fold: fold<T>, doc: doc::moddoc) -> doc::moddoc;
type fold_nmod<T> = fn~(fold: fold<T>, doc: doc::nmoddoc) -> doc::nmoddoc;
type fold_fn<T> = fn~(fold: fold<T>, doc: doc::fndoc) -> doc::fndoc;
type fold_const<T> = fn~(fold: fold<T>, doc: doc::constdoc) -> doc::constdoc;
type fold_enum<T> = fn~(fold: fold<T>, doc: doc::enumdoc) -> doc::enumdoc;
type fold_trait<T> = fn~(fold: fold<T>, doc: doc::traitdoc) -> doc::traitdoc;
type fold_impl<T> = fn~(fold: fold<T>, doc: doc::impldoc) -> doc::impldoc;
type fold_type<T> = fn~(fold: fold<T>, doc: doc::tydoc) -> doc::tydoc;

type t<T> = {
    ctxt: T,
    fold_doc: fold_doc<T>,
    fold_crate: fold_crate<T>,
    fold_item: fold_item<T>,
    fold_mod: fold_mod<T>,
    fold_nmod: fold_nmod<T>,
    fold_fn: fold_fn<T>,
    fold_const: fold_const<T>,
    fold_enum: fold_enum<T>,
    fold_trait: fold_trait<T>,
    fold_impl: fold_impl<T>,
    fold_type: fold_type<T>
};


// This exists because fn types don't infer correctly as record
// initializers, but they do as function arguments
fn mk_fold<T:copy>(
    ctxt: T,
    +fold_doc: fold_doc<T>,
    +fold_crate: fold_crate<T>,
    +fold_item: fold_item<T>,
    +fold_mod: fold_mod<T>,
    +fold_nmod: fold_nmod<T>,
    +fold_fn: fold_fn<T>,
    +fold_const: fold_const<T>,
    +fold_enum: fold_enum<T>,
    +fold_trait: fold_trait<T>,
    +fold_impl: fold_impl<T>,
    +fold_type: fold_type<T>
) -> fold<T> {
    fold({
        ctxt: ctxt,
        fold_doc: fold_doc,
        fold_crate: fold_crate,
        fold_item: fold_item,
        fold_mod: fold_mod,
        fold_nmod: fold_nmod,
        fold_fn: fold_fn,
        fold_const: fold_const,
        fold_enum: fold_enum,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        fold_type: fold_type
    })
}

fn default_any_fold<T:send copy>(ctxt: T) -> fold<T> {
    mk_fold(
        ctxt,
        |f, d| default_seq_fold_doc(f, d),
        |f, d| default_seq_fold_crate(f, d),
        |f, d| default_seq_fold_item(f, d),
        |f, d| default_any_fold_mod(f, d),
        |f, d| default_any_fold_nmod(f, d),
        |f, d| default_seq_fold_fn(f, d),
        |f, d| default_seq_fold_const(f, d),
        |f, d| default_seq_fold_enum(f, d),
        |f, d| default_seq_fold_trait(f, d),
        |f, d| default_seq_fold_impl(f, d),
        |f, d| default_seq_fold_type(f, d)
    )
}

fn default_seq_fold<T:copy>(ctxt: T) -> fold<T> {
    mk_fold(
        ctxt,
        |f, d| default_seq_fold_doc(f, d),
        |f, d| default_seq_fold_crate(f, d),
        |f, d| default_seq_fold_item(f, d),
        |f, d| default_seq_fold_mod(f, d),
        |f, d| default_seq_fold_nmod(f, d),
        |f, d| default_seq_fold_fn(f, d),
        |f, d| default_seq_fold_const(f, d),
        |f, d| default_seq_fold_enum(f, d),
        |f, d| default_seq_fold_trait(f, d),
        |f, d| default_seq_fold_impl(f, d),
        |f, d| default_seq_fold_type(f, d)
    )
}

fn default_par_fold<T:send copy>(ctxt: T) -> fold<T> {
    mk_fold(
        ctxt,
        |f, d| default_seq_fold_doc(f, d),
        |f, d| default_seq_fold_crate(f, d),
        |f, d| default_seq_fold_item(f, d),
        |f, d| default_par_fold_mod(f, d),
        |f, d| default_par_fold_nmod(f, d),
        |f, d| default_seq_fold_fn(f, d),
        |f, d| default_seq_fold_const(f, d),
        |f, d| default_seq_fold_enum(f, d),
        |f, d| default_seq_fold_trait(f, d),
        |f, d| default_seq_fold_impl(f, d),
        |f, d| default_seq_fold_type(f, d)
    )
}

fn default_seq_fold_doc<T>(fold: fold<T>, doc: doc::doc) -> doc::doc {
    doc::doc_({
        pages: do vec::map(doc.pages) |page| {
            alt page {
              doc::cratepage(doc) => {
                doc::cratepage(fold.fold_crate(fold, doc))
              }
              doc::itempage(doc) => {
                doc::itempage(fold_itemtag(fold, doc))
              }
            }
        }
        with *doc
    })
}

fn default_seq_fold_crate<T>(
    fold: fold<T>,
    doc: doc::cratedoc
) -> doc::cratedoc {
    {
        topmod: fold.fold_mod(fold, doc.topmod)
    }
}

fn default_seq_fold_item<T>(
    _fold: fold<T>,
    doc: doc::itemdoc
) -> doc::itemdoc {
    doc
}

fn default_any_fold_mod<T:send copy>(
    fold: fold<T>,
    doc: doc::moddoc
) -> doc::moddoc {
    doc::moddoc_({
        item: fold.fold_item(fold, doc.item),
        items: par::map(doc.items, |itemtag, copy fold| {
            fold_itemtag(fold, itemtag)
        })
        with *doc
    })
}

fn default_seq_fold_mod<T>(
    fold: fold<T>,
    doc: doc::moddoc
) -> doc::moddoc {
    doc::moddoc_({
        item: fold.fold_item(fold, doc.item),
        items: vec::map(doc.items, |itemtag| {
            fold_itemtag(fold, itemtag)
        })
        with *doc
    })
}

fn default_par_fold_mod<T:send copy>(
    fold: fold<T>,
    doc: doc::moddoc
) -> doc::moddoc {
    doc::moddoc_({
        item: fold.fold_item(fold, doc.item),
        items: par::map(doc.items, |itemtag, copy fold| {
            fold_itemtag(fold, itemtag)
        })
        with *doc
    })
}

fn default_any_fold_nmod<T:send copy>(
    fold: fold<T>,
    doc: doc::nmoddoc
) -> doc::nmoddoc {
    {
        item: fold.fold_item(fold, doc.item),
        fns: par::map(doc.fns, |fndoc, copy fold| {
            fold.fold_fn(fold, fndoc)
        })
        with doc
    }
}

fn default_seq_fold_nmod<T>(
    fold: fold<T>,
    doc: doc::nmoddoc
) -> doc::nmoddoc {
    {
        item: fold.fold_item(fold, doc.item),
        fns: vec::map(doc.fns, |fndoc| {
            fold.fold_fn(fold, fndoc)
        })
        with doc
    }
}

fn default_par_fold_nmod<T:send copy>(
    fold: fold<T>,
    doc: doc::nmoddoc
) -> doc::nmoddoc {
    {
        item: fold.fold_item(fold, doc.item),
        fns: par::map(doc.fns, |fndoc, copy fold| {
            fold.fold_fn(fold, fndoc)
        })
        with doc
    }
}

fn fold_itemtag<T>(fold: fold<T>, doc: doc::itemtag) -> doc::itemtag {
    alt doc {
      doc::modtag(moddoc) => {
        doc::modtag(fold.fold_mod(fold, moddoc))
      }
      doc::nmodtag(nmoddoc) => {
        doc::nmodtag(fold.fold_nmod(fold, nmoddoc))
      }
      doc::fntag(fndoc) => {
        doc::fntag(fold.fold_fn(fold, fndoc))
      }
      doc::consttag(constdoc) => {
        doc::consttag(fold.fold_const(fold, constdoc))
      }
      doc::enumtag(enumdoc) => {
        doc::enumtag(fold.fold_enum(fold, enumdoc))
      }
      doc::traittag(traitdoc) => {
        doc::traittag(fold.fold_trait(fold, traitdoc))
      }
      doc::impltag(impldoc) => {
        doc::impltag(fold.fold_impl(fold, impldoc))
      }
      doc::tytag(tydoc) => {
        doc::tytag(fold.fold_type(fold, tydoc))
      }
    }
}

fn default_seq_fold_fn<T>(
    fold: fold<T>,
    doc: doc::fndoc
) -> doc::fndoc {
    {
        item: fold.fold_item(fold, doc.item)
        with doc
    }
}

fn default_seq_fold_const<T>(
    fold: fold<T>,
    doc: doc::constdoc
) -> doc::constdoc {
    {
        item: fold.fold_item(fold, doc.item)
        with doc
    }
}

fn default_seq_fold_enum<T>(
    fold: fold<T>,
    doc: doc::enumdoc
) -> doc::enumdoc {
    {
        item: fold.fold_item(fold, doc.item)
        with doc
    }
}

fn default_seq_fold_trait<T>(
    fold: fold<T>,
    doc: doc::traitdoc
) -> doc::traitdoc {
    {
        item: fold.fold_item(fold, doc.item)
        with doc
    }
}

fn default_seq_fold_impl<T>(
    fold: fold<T>,
    doc: doc::impldoc
) -> doc::impldoc {
    {
        item: fold.fold_item(fold, doc.item)
        with doc
    }
}

fn default_seq_fold_type<T>(
    fold: fold<T>,
    doc: doc::tydoc
) -> doc::tydoc {
    {
        item: fold.fold_item(fold, doc.item)
        with doc
    }
}

#[test]
fn default_fold_should_produce_same_doc() {
    let source = ~"mod a { fn b() { } mod c { fn d() { } } }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_seq_fold(());
    let folded = fld.fold_doc(fld, doc);
    assert doc == folded;
}

#[test]
fn default_fold_should_produce_same_consts() {
    let source = ~"const a: int = 0;";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_seq_fold(());
    let folded = fld.fold_doc(fld, doc);
    assert doc == folded;
}

#[test]
fn default_fold_should_produce_same_enums() {
    let source = ~"enum a { b }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_seq_fold(());
    let folded = fld.fold_doc(fld, doc);
    assert doc == folded;
}

#[test]
fn default_parallel_fold_should_produce_same_doc() {
    let source = ~"mod a { fn b() { } mod c { fn d() { } } }";
    let ast = parse::from_str(source);
    let doc = extract::extract(ast, ~"");
    let fld = default_par_fold(());
    let folded = fld.fold_doc(fld, doc);
    assert doc == folded;
}
