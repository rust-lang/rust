// A signature is a string representation of an item's type signature, excluding
// any body. It also includes ids for any defs or refs in the signature. For
// example:
//
// ```
// fn foo(x: String) {
//     println!("{}", x);
// }
// ```
// The signature string is something like "fn foo(x: String) {}" and the signature
// will have defs for `foo` and `x` and a ref for `String`.
//
// All signature text should parse in the correct context (i.e., in a module or
// impl, etc.). Clients may want to trim trailing `{}` or `;`. The text of a
// signature is not guaranteed to be stable (it may improve or change as the
// syntax changes, or whitespace or punctuation may change). It is also likely
// not to be pretty - no attempt is made to prettify the text. It is recommended
// that clients run the text through Rustfmt.
//
// This module generates Signatures for items by walking the AST and looking up
// references.
//
// Signatures do not include visibility info. I'm not sure if this is a feature
// or an ommission (FIXME).
//
// FIXME where clauses need implementing, defs/refs in generics are mostly missing.

use crate::{id_from_def_id, id_from_node_id, SaveContext};

use rls_data::{SigElement, Signature};

use rustc::hir::def::{Res, DefKind};
use syntax::ast::{self, NodeId};
use syntax::print::pprust;


pub fn item_signature(item: &ast::Item, scx: &SaveContext<'_, '_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    item.make(0, None, scx).ok()
}

pub fn foreign_item_signature(
    item: &ast::ForeignItem,
    scx: &SaveContext<'_, '_>
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    item.make(0, None, scx).ok()
}

/// Signature for a struct or tuple field declaration.
/// Does not include a trailing comma.
pub fn field_signature(field: &ast::StructField, scx: &SaveContext<'_, '_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    field.make(0, None, scx).ok()
}

/// Does not include a trailing comma.
pub fn variant_signature(variant: &ast::Variant, scx: &SaveContext<'_, '_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    variant.node.make(0, None, scx).ok()
}

pub fn method_signature(
    id: NodeId,
    ident: ast::Ident,
    generics: &ast::Generics,
    m: &ast::MethodSig,
    scx: &SaveContext<'_, '_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    make_method_signature(id, ident, generics, m, scx).ok()
}

pub fn assoc_const_signature(
    id: NodeId,
    ident: ast::Name,
    ty: &ast::Ty,
    default: Option<&ast::Expr>,
    scx: &SaveContext<'_, '_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    make_assoc_const_signature(id, ident, ty, default, scx).ok()
}

pub fn assoc_type_signature(
    id: NodeId,
    ident: ast::Ident,
    bounds: Option<&ast::GenericBounds>,
    default: Option<&ast::Ty>,
    scx: &SaveContext<'_, '_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    make_assoc_type_signature(id, ident, bounds, default, scx).ok()
}

type Result = std::result::Result<Signature, &'static str>;

trait Sig {
    fn make(&self, offset: usize, id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result;
}

fn extend_sig(
    mut sig: Signature,
    text: String,
    defs: Vec<SigElement>,
    refs: Vec<SigElement>,
) -> Signature {
    sig.text = text;
    sig.defs.extend(defs.into_iter());
    sig.refs.extend(refs.into_iter());
    sig
}

fn replace_text(mut sig: Signature, text: String) -> Signature {
    sig.text = text;
    sig
}

fn merge_sigs(text: String, sigs: Vec<Signature>) -> Signature {
    let mut result = Signature {
        text,
        defs: vec![],
        refs: vec![],
    };

    let (defs, refs): (Vec<_>, Vec<_>) = sigs.into_iter().map(|s| (s.defs, s.refs)).unzip();

    result
        .defs
        .extend(defs.into_iter().flat_map(|ds| ds.into_iter()));
    result
        .refs
        .extend(refs.into_iter().flat_map(|rs| rs.into_iter()));

    result
}

fn text_sig(text: String) -> Signature {
    Signature {
        text,
        defs: vec![],
        refs: vec![],
    }
}

impl Sig for ast::Ty {
    fn make(&self, offset: usize, _parent_id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        let id = Some(self.id);
        match self.node {
            ast::TyKind::Slice(ref ty) => {
                let nested = ty.make(offset + 1, id, scx)?;
                let text = format!("[{}]", nested.text);
                Ok(replace_text(nested, text))
            }
            ast::TyKind::Ptr(ref mt) => {
                let prefix = match mt.mutbl {
                    ast::Mutability::Mutable => "*mut ",
                    ast::Mutability::Immutable => "*const ",
                };
                let nested = mt.ty.make(offset + prefix.len(), id, scx)?;
                let text = format!("{}{}", prefix, nested.text);
                Ok(replace_text(nested, text))
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                let mut prefix = "&".to_owned();
                if let &Some(ref l) = lifetime {
                    prefix.push_str(&l.ident.to_string());
                    prefix.push(' ');
                }
                if let ast::Mutability::Mutable = mt.mutbl {
                    prefix.push_str("mut ");
                };

                let nested = mt.ty.make(offset + prefix.len(), id, scx)?;
                let text = format!("{}{}", prefix, nested.text);
                Ok(replace_text(nested, text))
            }
            ast::TyKind::Never => Ok(text_sig("!".to_owned())),
            ast::TyKind::CVarArgs => Ok(text_sig("...".to_owned())),
            ast::TyKind::Tup(ref ts) => {
                let mut text = "(".to_owned();
                let mut defs = vec![];
                let mut refs = vec![];
                for t in ts {
                    let nested = t.make(offset + text.len(), id, scx)?;
                    text.push_str(&nested.text);
                    text.push(',');
                    defs.extend(nested.defs.into_iter());
                    refs.extend(nested.refs.into_iter());
                }
                text.push(')');
                Ok(Signature { text, defs, refs })
            }
            ast::TyKind::Paren(ref ty) => {
                let nested = ty.make(offset + 1, id, scx)?;
                let text = format!("({})", nested.text);
                Ok(replace_text(nested, text))
            }
            ast::TyKind::BareFn(ref f) => {
                let mut text = String::new();
                if !f.generic_params.is_empty() {
                    // FIXME defs, bounds on lifetimes
                    text.push_str("for<");
                    text.push_str(&f.generic_params
                        .iter()
                        .filter_map(|param| match param.kind {
                            ast::GenericParamKind::Lifetime { .. } => {
                                Some(param.ident.to_string())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(", "));
                    text.push('>');
                }

                if f.unsafety == ast::Unsafety::Unsafe {
                    text.push_str("unsafe ");
                }
                if f.abi != rustc_target::spec::abi::Abi::Rust {
                    text.push_str("extern");
                    text.push_str(&f.abi.to_string());
                    text.push(' ');
                }
                text.push_str("fn(");

                let mut defs = vec![];
                let mut refs = vec![];
                for i in &f.decl.inputs {
                    let nested = i.ty.make(offset + text.len(), Some(i.id), scx)?;
                    text.push_str(&nested.text);
                    text.push(',');
                    defs.extend(nested.defs.into_iter());
                    refs.extend(nested.refs.into_iter());
                }
                text.push(')');
                if let ast::FunctionRetTy::Ty(ref t) = f.decl.output {
                    text.push_str(" -> ");
                    let nested = t.make(offset + text.len(), None, scx)?;
                    text.push_str(&nested.text);
                    text.push(',');
                    defs.extend(nested.defs.into_iter());
                    refs.extend(nested.refs.into_iter());
                }

                Ok(Signature { text, defs, refs })
            }
            ast::TyKind::Path(None, ref path) => path.make(offset, id, scx),
            ast::TyKind::Path(Some(ref qself), ref path) => {
                let nested_ty = qself.ty.make(offset + 1, id, scx)?;
                let prefix = if qself.position == 0 {
                    format!("<{}>::", nested_ty.text)
                } else if qself.position == 1 {
                    let first = pprust::path_segment_to_string(&path.segments[0]);
                    format!("<{} as {}>::", nested_ty.text, first)
                } else {
                    // FIXME handle path instead of elipses.
                    format!("<{} as ...>::", nested_ty.text)
                };

                let name = pprust::path_segment_to_string(path.segments.last().ok_or("Bad path")?);
                let res = scx.get_path_res(id.ok_or("Missing id for Path")?);
                let id = id_from_def_id(res.def_id());
                if path.segments.len() - qself.position == 1 {
                    let start = offset + prefix.len();
                    let end = start + name.len();

                    Ok(Signature {
                        text: prefix + &name,
                        defs: vec![],
                        refs: vec![SigElement { id, start, end }],
                    })
                } else {
                    let start = offset + prefix.len() + 5;
                    let end = start + name.len();
                    // FIXME should put the proper path in there, not elipses.
                    Ok(Signature {
                        text: prefix + "...::" + &name,
                        defs: vec![],
                        refs: vec![SigElement { id, start, end }],
                    })
                }
            }
            ast::TyKind::TraitObject(ref bounds, ..) => {
                // FIXME recurse into bounds
                let nested = pprust::bounds_to_string(bounds);
                Ok(text_sig(nested))
            }
            ast::TyKind::ImplTrait(_, ref bounds) => {
                // FIXME recurse into bounds
                let nested = pprust::bounds_to_string(bounds);
                Ok(text_sig(format!("impl {}", nested)))
            }
            ast::TyKind::Array(ref ty, ref v) => {
                let nested_ty = ty.make(offset + 1, id, scx)?;
                let expr = pprust::expr_to_string(&v.value).replace('\n', " ");
                let text = format!("[{}; {}]", nested_ty.text, expr);
                Ok(replace_text(nested_ty, text))
            }
            ast::TyKind::Typeof(_) |
            ast::TyKind::Infer |
            ast::TyKind::Err |
            ast::TyKind::ImplicitSelf |
            ast::TyKind::Mac(_) => Err("Ty"),
        }
    }
}

impl Sig for ast::Item {
    fn make(&self, offset: usize, _parent_id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        let id = Some(self.id);

        match self.node {
            ast::ItemKind::Static(ref ty, m, ref expr) => {
                let mut text = "static ".to_owned();
                if m == ast::Mutability::Mutable {
                    text.push_str("mut ");
                }
                let name = self.ident.to_string();
                let defs = vec![
                    SigElement {
                        id: id_from_node_id(self.id, scx),
                        start: offset + text.len(),
                        end: offset + text.len() + name.len(),
                    },
                ];
                text.push_str(&name);
                text.push_str(": ");

                let ty = ty.make(offset + text.len(), id, scx)?;
                text.push_str(&ty.text);
                text.push_str(" = ");

                let expr = pprust::expr_to_string(expr).replace('\n', " ");
                text.push_str(&expr);
                text.push(';');

                Ok(extend_sig(ty, text, defs, vec![]))
            }
            ast::ItemKind::Const(ref ty, ref expr) => {
                let mut text = "const ".to_owned();
                let name = self.ident.to_string();
                let defs = vec![
                    SigElement {
                        id: id_from_node_id(self.id, scx),
                        start: offset + text.len(),
                        end: offset + text.len() + name.len(),
                    },
                ];
                text.push_str(&name);
                text.push_str(": ");

                let ty = ty.make(offset + text.len(), id, scx)?;
                text.push_str(&ty.text);
                text.push_str(" = ");

                let expr = pprust::expr_to_string(expr).replace('\n', " ");
                text.push_str(&expr);
                text.push(';');

                Ok(extend_sig(ty, text, defs, vec![]))
            }
            ast::ItemKind::Fn(ref decl, header, ref generics, _) => {
                let mut text = String::new();
                if header.constness.node == ast::Constness::Const {
                    text.push_str("const ");
                }
                if header.asyncness.node.is_async() {
                    text.push_str("async ");
                }
                if header.unsafety == ast::Unsafety::Unsafe {
                    text.push_str("unsafe ");
                }
                if header.abi != rustc_target::spec::abi::Abi::Rust {
                    text.push_str("extern");
                    text.push_str(&header.abi.to_string());
                    text.push(' ');
                }
                text.push_str("fn ");

                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;

                sig.text.push('(');
                for i in &decl.inputs {
                    // FIXME should descend into patterns to add defs.
                    sig.text.push_str(&pprust::pat_to_string(&i.pat));
                    sig.text.push_str(": ");
                    let nested = i.ty.make(offset + sig.text.len(), Some(i.id), scx)?;
                    sig.text.push_str(&nested.text);
                    sig.text.push(',');
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push(')');

                if let ast::FunctionRetTy::Ty(ref t) = decl.output {
                    sig.text.push_str(" -> ");
                    let nested = t.make(offset + sig.text.len(), None, scx)?;
                    sig.text.push_str(&nested.text);
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push_str(" {}");

                Ok(sig)
            }
            ast::ItemKind::Mod(ref _mod) => {
                let mut text = "mod ".to_owned();
                let name = self.ident.to_string();
                let defs = vec![
                    SigElement {
                        id: id_from_node_id(self.id, scx),
                        start: offset + text.len(),
                        end: offset + text.len() + name.len(),
                    },
                ];
                text.push_str(&name);
                // Could be either `mod foo;` or `mod foo { ... }`, but we'll just pick one.
                text.push(';');

                Ok(Signature {
                    text,
                    defs,
                    refs: vec![],
                })
            }
            ast::ItemKind::Existential(ref bounds, ref generics) => {
                let text = "existential type ".to_owned();
                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;

                if !bounds.is_empty() {
                    sig.text.push_str(": ");
                    sig.text.push_str(&pprust::bounds_to_string(bounds));
                }
                sig.text.push(';');

                Ok(sig)
            }
            ast::ItemKind::Ty(ref ty, ref generics) => {
                let text = "type ".to_owned();
                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;

                sig.text.push_str(" = ");
                let ty = ty.make(offset + sig.text.len(), id, scx)?;
                sig.text.push_str(&ty.text);
                sig.text.push(';');

                Ok(merge_sigs(sig.text.clone(), vec![sig, ty]))
            }
            ast::ItemKind::Enum(_, ref generics) => {
                let text = "enum ".to_owned();
                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;
                sig.text.push_str(" {}");
                Ok(sig)
            }
            ast::ItemKind::Struct(_, ref generics) => {
                let text = "struct ".to_owned();
                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;
                sig.text.push_str(" {}");
                Ok(sig)
            }
            ast::ItemKind::Union(_, ref generics) => {
                let text = "union ".to_owned();
                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;
                sig.text.push_str(" {}");
                Ok(sig)
            }
            ast::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, _) => {
                let mut text = String::new();

                if is_auto == ast::IsAuto::Yes {
                    text.push_str("auto ");
                }

                if unsafety == ast::Unsafety::Unsafe {
                    text.push_str("unsafe ");
                }
                text.push_str("trait ");
                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;

                if !bounds.is_empty() {
                    sig.text.push_str(": ");
                    sig.text.push_str(&pprust::bounds_to_string(bounds));
                }
                // FIXME where clause
                sig.text.push_str(" {}");

                Ok(sig)
            }
            ast::ItemKind::TraitAlias(ref generics, ref bounds) => {
                let mut text = String::new();
                text.push_str("trait ");
                let mut sig = name_and_generics(text,
                                                offset,
                                                generics,
                                                self.id,
                                                self.ident,
                                                scx)?;

                if !bounds.is_empty() {
                    sig.text.push_str(" = ");
                    sig.text.push_str(&pprust::bounds_to_string(bounds));
                }
                // FIXME where clause
                sig.text.push_str(";");

                Ok(sig)
            }
            ast::ItemKind::Impl(
                unsafety,
                polarity,
                defaultness,
                ref generics,
                ref opt_trait,
                ref ty,
                _,
            ) => {
                let mut text = String::new();
                if let ast::Defaultness::Default = defaultness {
                    text.push_str("default ");
                }
                if unsafety == ast::Unsafety::Unsafe {
                    text.push_str("unsafe ");
                }
                text.push_str("impl");

                let generics_sig = generics.make(offset + text.len(), id, scx)?;
                text.push_str(&generics_sig.text);

                text.push(' ');

                let trait_sig = if let Some(ref t) = *opt_trait {
                    if polarity == ast::ImplPolarity::Negative {
                        text.push('!');
                    }
                    let trait_sig = t.path.make(offset + text.len(), id, scx)?;
                    text.push_str(&trait_sig.text);
                    text.push_str(" for ");
                    trait_sig
                } else {
                    text_sig(String::new())
                };

                let ty_sig = ty.make(offset + text.len(), id, scx)?;
                text.push_str(&ty_sig.text);

                text.push_str(" {}");

                Ok(merge_sigs(text, vec![generics_sig, trait_sig, ty_sig]))

                // FIXME where clause
            }
            ast::ItemKind::ForeignMod(_) => Err("extern mod"),
            ast::ItemKind::GlobalAsm(_) => Err("glboal asm"),
            ast::ItemKind::ExternCrate(_) => Err("extern crate"),
            // FIXME should implement this (e.g., pub use).
            ast::ItemKind::Use(_) => Err("import"),
            ast::ItemKind::Mac(..) | ast::ItemKind::MacroDef(_) => Err("Macro"),
        }
    }
}

impl Sig for ast::Path {
    fn make(&self, offset: usize, id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        let res = scx.get_path_res(id.ok_or("Missing id for Path")?);

        let (name, start, end) = match res {
            Res::PrimTy(..) | Res::SelfTy(..) | Res::Err => {
                return Ok(Signature {
                    text: pprust::path_to_string(self),
                    defs: vec![],
                    refs: vec![],
                })
            }
            Res::Def(DefKind::AssocConst, _)
            | Res::Def(DefKind::Variant, _)
            | Res::Def(DefKind::Ctor(..), _) => {
                let len = self.segments.len();
                if len < 2 {
                    return Err("Bad path");
                }
                // FIXME: really we should descend into the generics here and add SigElements for
                // them.
                // FIXME: would be nice to have a def for the first path segment.
                let seg1 = pprust::path_segment_to_string(&self.segments[len - 2]);
                let seg2 = pprust::path_segment_to_string(&self.segments[len - 1]);
                let start = offset + seg1.len() + 2;
                (format!("{}::{}", seg1, seg2), start, start + seg2.len())
            }
            _ => {
                let name = pprust::path_segment_to_string(self.segments.last().ok_or("Bad path")?);
                let end = offset + name.len();
                (name, offset, end)
            }
        };

        let id = id_from_def_id(res.def_id());
        Ok(Signature {
            text: name,
            defs: vec![],
            refs: vec![SigElement { id, start, end }],
        })
    }
}

// This does not cover the where clause, which must be processed separately.
impl Sig for ast::Generics {
    fn make(&self, offset: usize, _parent_id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        if self.params.is_empty() {
            return Ok(text_sig(String::new()));
        }

        let mut text = "<".to_owned();

        let mut defs = Vec::with_capacity(self.params.len());
        for param in &self.params {
            let mut param_text = String::new();
            if let ast::GenericParamKind::Const { .. } = param.kind {
                param_text.push_str("const ");
            }
            param_text.push_str(&param.ident.as_str());
            defs.push(SigElement {
                id: id_from_node_id(param.id, scx),
                start: offset + text.len(),
                end: offset + text.len() + param_text.as_str().len(),
            });
            if let ast::GenericParamKind::Const { ref ty } = param.kind {
                param_text.push_str(": ");
                param_text.push_str(&pprust::ty_to_string(&ty));
            }
            if !param.bounds.is_empty() {
                param_text.push_str(": ");
                match param.kind {
                    ast::GenericParamKind::Lifetime { .. } => {
                        let bounds = param.bounds.iter()
                            .map(|bound| match bound {
                                ast::GenericBound::Outlives(lt) => lt.ident.to_string(),
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>()
                            .join(" + ");
                        param_text.push_str(&bounds);
                        // FIXME add lifetime bounds refs.
                    }
                    ast::GenericParamKind::Type { .. } => {
                        param_text.push_str(&pprust::bounds_to_string(&param.bounds));
                        // FIXME descend properly into bounds.
                    }
                    ast::GenericParamKind::Const { .. } => {
                        // Const generics cannot contain bounds.
                    }
                }
            }
            text.push_str(&param_text);
            text.push(',');
        }

        text.push('>');
        Ok(Signature {
            text,
            defs,
            refs: vec![],
        })
    }
}

impl Sig for ast::StructField {
    fn make(&self, offset: usize, _parent_id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        let mut text = String::new();
        let mut defs = None;
        if let Some(ident) = self.ident {
            text.push_str(&ident.to_string());
            defs = Some(SigElement {
                id: id_from_node_id(self.id, scx),
                start: offset,
                end: offset + text.len(),
            });
            text.push_str(": ");
        }

        let mut ty_sig = self.ty.make(offset + text.len(), Some(self.id), scx)?;
        text.push_str(&ty_sig.text);
        ty_sig.text = text;
        ty_sig.defs.extend(defs.into_iter());
        Ok(ty_sig)
    }
}


impl Sig for ast::Variant_ {
    fn make(&self, offset: usize, parent_id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        let mut text = self.ident.to_string();
        match self.data {
            ast::VariantData::Struct(ref fields, r) => {
                let id = parent_id.unwrap();
                let name_def = SigElement {
                    id: id_from_node_id(id, scx),
                    start: offset,
                    end: offset + text.len(),
                };
                text.push_str(" { ");
                let mut defs = vec![name_def];
                let mut refs = vec![];
                if r {
                    text.push_str("/* parse error */ ");
                } else {
                    for f in fields {
                        let field_sig = f.make(offset + text.len(), Some(id), scx)?;
                        text.push_str(&field_sig.text);
                        text.push_str(", ");
                        defs.extend(field_sig.defs.into_iter());
                        refs.extend(field_sig.refs.into_iter());
                    }
                }
                text.push('}');
                Ok(Signature { text, defs, refs })
            }
            ast::VariantData::Tuple(ref fields, id) => {
                let name_def = SigElement {
                    id: id_from_node_id(id, scx),
                    start: offset,
                    end: offset + text.len(),
                };
                text.push('(');
                let mut defs = vec![name_def];
                let mut refs = vec![];
                for f in fields {
                    let field_sig = f.make(offset + text.len(), Some(id), scx)?;
                    text.push_str(&field_sig.text);
                    text.push_str(", ");
                    defs.extend(field_sig.defs.into_iter());
                    refs.extend(field_sig.refs.into_iter());
                }
                text.push(')');
                Ok(Signature { text, defs, refs })
            }
            ast::VariantData::Unit(id) => {
                let name_def = SigElement {
                    id: id_from_node_id(id, scx),
                    start: offset,
                    end: offset + text.len(),
                };
                Ok(Signature {
                    text,
                    defs: vec![name_def],
                    refs: vec![],
                })
            }
        }
    }
}

impl Sig for ast::ForeignItem {
    fn make(&self, offset: usize, _parent_id: Option<NodeId>, scx: &SaveContext<'_, '_>) -> Result {
        let id = Some(self.id);
        match self.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                let mut text = String::new();
                text.push_str("fn ");

                let mut sig = name_and_generics(text, offset, generics, self.id, self.ident, scx)?;

                sig.text.push('(');
                for i in &decl.inputs {
                    // FIXME should descend into patterns to add defs.
                    sig.text.push_str(&pprust::pat_to_string(&i.pat));
                    sig.text.push_str(": ");
                    let nested = i.ty.make(offset + sig.text.len(), Some(i.id), scx)?;
                    sig.text.push_str(&nested.text);
                    sig.text.push(',');
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push(')');

                if let ast::FunctionRetTy::Ty(ref t) = decl.output {
                    sig.text.push_str(" -> ");
                    let nested = t.make(offset + sig.text.len(), None, scx)?;
                    sig.text.push_str(&nested.text);
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push(';');

                Ok(sig)
            }
            ast::ForeignItemKind::Static(ref ty, m) => {
                let mut text = "static ".to_owned();
                if m == ast::Mutability::Mutable {
                    text.push_str("mut ");
                }
                let name = self.ident.to_string();
                let defs = vec![
                    SigElement {
                        id: id_from_node_id(self.id, scx),
                        start: offset + text.len(),
                        end: offset + text.len() + name.len(),
                    },
                ];
                text.push_str(&name);
                text.push_str(": ");

                let ty_sig = ty.make(offset + text.len(), id, scx)?;
                text.push(';');

                Ok(extend_sig(ty_sig, text, defs, vec![]))
            }
            ast::ForeignItemKind::Ty => {
                let mut text = "type ".to_owned();
                let name = self.ident.to_string();
                let defs = vec![
                    SigElement {
                        id: id_from_node_id(self.id, scx),
                        start: offset + text.len(),
                        end: offset + text.len() + name.len(),
                    },
                ];
                text.push_str(&name);
                text.push(';');

                Ok(Signature {
                    text: text,
                    defs: defs,
                    refs: vec![],
                })
            }
            ast::ForeignItemKind::Macro(..) => Err("macro"),
        }
    }
}

fn name_and_generics(
    mut text: String,
    offset: usize,
    generics: &ast::Generics,
    id: NodeId,
    name: ast::Ident,
    scx: &SaveContext<'_, '_>,
) -> Result {
    let name = name.to_string();
    let def = SigElement {
        id: id_from_node_id(id, scx),
        start: offset + text.len(),
        end: offset + text.len() + name.len(),
    };
    text.push_str(&name);
    let generics: Signature = generics.make(offset + text.len(), Some(id), scx)?;
    // FIXME where clause
    let text = format!("{}{}", text, generics.text);
    Ok(extend_sig(generics, text, vec![def], vec![]))
}


fn make_assoc_type_signature(
    id: NodeId,
    ident: ast::Ident,
    bounds: Option<&ast::GenericBounds>,
    default: Option<&ast::Ty>,
    scx: &SaveContext<'_, '_>,
) -> Result {
    let mut text = "type ".to_owned();
    let name = ident.to_string();
    let mut defs = vec![
        SigElement {
            id: id_from_node_id(id, scx),
            start: text.len(),
            end: text.len() + name.len(),
        },
    ];
    let mut refs = vec![];
    text.push_str(&name);
    if let Some(bounds) = bounds {
        text.push_str(": ");
        // FIXME should descend into bounds
        text.push_str(&pprust::bounds_to_string(bounds));
    }
    if let Some(default) = default {
        text.push_str(" = ");
        let ty_sig = default.make(text.len(), Some(id), scx)?;
        text.push_str(&ty_sig.text);
        defs.extend(ty_sig.defs.into_iter());
        refs.extend(ty_sig.refs.into_iter());
    }
    text.push(';');
    Ok(Signature { text, defs, refs })
}

fn make_assoc_const_signature(
    id: NodeId,
    ident: ast::Name,
    ty: &ast::Ty,
    default: Option<&ast::Expr>,
    scx: &SaveContext<'_, '_>,
) -> Result {
    let mut text = "const ".to_owned();
    let name = ident.to_string();
    let mut defs = vec![
        SigElement {
            id: id_from_node_id(id, scx),
            start: text.len(),
            end: text.len() + name.len(),
        },
    ];
    let mut refs = vec![];
    text.push_str(&name);
    text.push_str(": ");

    let ty_sig = ty.make(text.len(), Some(id), scx)?;
    text.push_str(&ty_sig.text);
    defs.extend(ty_sig.defs.into_iter());
    refs.extend(ty_sig.refs.into_iter());

    if let Some(default) = default {
        text.push_str(" = ");
        text.push_str(&pprust::expr_to_string(default));
    }
    text.push(';');
    Ok(Signature { text, defs, refs })
}

fn make_method_signature(
    id: NodeId,
    ident: ast::Ident,
    generics: &ast::Generics,
    m: &ast::MethodSig,
    scx: &SaveContext<'_, '_>,
) -> Result {
    // FIXME code dup with function signature
    let mut text = String::new();
    if m.header.constness.node == ast::Constness::Const {
        text.push_str("const ");
    }
    if m.header.asyncness.node.is_async() {
        text.push_str("async ");
    }
    if m.header.unsafety == ast::Unsafety::Unsafe {
        text.push_str("unsafe ");
    }
    if m.header.abi != rustc_target::spec::abi::Abi::Rust {
        text.push_str("extern");
        text.push_str(&m.header.abi.to_string());
        text.push(' ');
    }
    text.push_str("fn ");

    let mut sig = name_and_generics(text, 0, generics, id, ident, scx)?;

    sig.text.push('(');
    for i in &m.decl.inputs {
        // FIXME should descend into patterns to add defs.
        sig.text.push_str(&pprust::pat_to_string(&i.pat));
        sig.text.push_str(": ");
        let nested = i.ty.make(sig.text.len(), Some(i.id), scx)?;
        sig.text.push_str(&nested.text);
        sig.text.push(',');
        sig.defs.extend(nested.defs.into_iter());
        sig.refs.extend(nested.refs.into_iter());
    }
    sig.text.push(')');

    if let ast::FunctionRetTy::Ty(ref t) = m.decl.output {
        sig.text.push_str(" -> ");
        let nested = t.make(sig.text.len(), None, scx)?;
        sig.text.push_str(&nested.text);
        sig.defs.extend(nested.defs.into_iter());
        sig.refs.extend(nested.refs.into_iter());
    }
    sig.text.push_str(" {}");

    Ok(sig)
}
