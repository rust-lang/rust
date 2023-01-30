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
// or an omission (FIXME).
//
// FIXME where clauses need implementing, defs/refs in generics are mostly missing.

use crate::{id_from_def_id, SaveContext};

use rls_data::{SigElement, Signature};

use rustc_ast::Mutability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir_pretty::id_to_string;
use rustc_hir_pretty::{bounds_to_string, path_segment_to_string, path_to_string, ty_to_string};
use rustc_span::def_id::LocalDefId;
use rustc_span::symbol::{Ident, Symbol};

pub fn item_signature(item: &hir::Item<'_>, scx: &SaveContext<'_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    item.make(0, None, scx).ok()
}

pub fn foreign_item_signature(
    item: &hir::ForeignItem<'_>,
    scx: &SaveContext<'_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    item.make(0, None, scx).ok()
}

/// Signature for a struct or tuple field declaration.
/// Does not include a trailing comma.
pub fn field_signature(field: &hir::FieldDef<'_>, scx: &SaveContext<'_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    field.make(0, None, scx).ok()
}

/// Does not include a trailing comma.
pub fn variant_signature(variant: &hir::Variant<'_>, scx: &SaveContext<'_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    variant.make(0, None, scx).ok()
}

pub fn method_signature(
    id: hir::OwnerId,
    ident: Ident,
    generics: &hir::Generics<'_>,
    m: &hir::FnSig<'_>,
    scx: &SaveContext<'_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    make_method_signature(id, ident, generics, m, scx).ok()
}

pub fn assoc_const_signature(
    id: hir::OwnerId,
    ident: Symbol,
    ty: &hir::Ty<'_>,
    default: Option<&hir::Expr<'_>>,
    scx: &SaveContext<'_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    make_assoc_const_signature(id, ident, ty, default, scx).ok()
}

pub fn assoc_type_signature(
    id: hir::OwnerId,
    ident: Ident,
    bounds: Option<hir::GenericBounds<'_>>,
    default: Option<&hir::Ty<'_>>,
    scx: &SaveContext<'_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    make_assoc_type_signature(id, ident, bounds, default, scx).ok()
}

type Result = std::result::Result<Signature, &'static str>;

trait Sig {
    type Parent;
    fn make(&self, offset: usize, id: Option<Self::Parent>, scx: &SaveContext<'_>) -> Result;
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
    let mut result = Signature { text, defs: vec![], refs: vec![] };

    let (defs, refs): (Vec<_>, Vec<_>) = sigs.into_iter().map(|s| (s.defs, s.refs)).unzip();

    result.defs.extend(defs.into_iter().flat_map(|ds| ds.into_iter()));
    result.refs.extend(refs.into_iter().flat_map(|rs| rs.into_iter()));

    result
}

fn text_sig(text: String) -> Signature {
    Signature { text, defs: vec![], refs: vec![] }
}

impl<'hir> Sig for hir::Ty<'hir> {
    type Parent = hir::HirId;
    fn make(&self, offset: usize, _parent_id: Option<hir::HirId>, scx: &SaveContext<'_>) -> Result {
        let id = Some(self.hir_id);
        match self.kind {
            hir::TyKind::Slice(ref ty) => {
                let nested = ty.make(offset + 1, id, scx)?;
                let text = format!("[{}]", nested.text);
                Ok(replace_text(nested, text))
            }
            hir::TyKind::Ptr(ref mt) => {
                let prefix = match mt.mutbl {
                    hir::Mutability::Mut => "*mut ",
                    hir::Mutability::Not => "*const ",
                };
                let nested = mt.ty.make(offset + prefix.len(), id, scx)?;
                let text = format!("{}{}", prefix, nested.text);
                Ok(replace_text(nested, text))
            }
            hir::TyKind::Ref(ref lifetime, ref mt) => {
                let mut prefix = "&".to_owned();
                prefix.push_str(&lifetime.ident.to_string());
                prefix.push(' ');
                if mt.mutbl.is_mut() {
                    prefix.push_str("mut ");
                };

                let nested = mt.ty.make(offset + prefix.len(), id, scx)?;
                let text = format!("{}{}", prefix, nested.text);
                Ok(replace_text(nested, text))
            }
            hir::TyKind::Never => Ok(text_sig("!".to_owned())),
            hir::TyKind::Tup(ts) => {
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
            hir::TyKind::BareFn(ref f) => {
                let mut text = String::new();
                if !f.generic_params.is_empty() {
                    // FIXME defs, bounds on lifetimes
                    text.push_str("for<");
                    text.push_str(
                        &f.generic_params
                            .iter()
                            .filter_map(|param| match param.kind {
                                hir::GenericParamKind::Lifetime { .. } => {
                                    Some(param.name.ident().to_string())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    text.push('>');
                }

                if let hir::Unsafety::Unsafe = f.unsafety {
                    text.push_str("unsafe ");
                }
                text.push_str("fn(");

                let mut defs = vec![];
                let mut refs = vec![];
                for i in f.decl.inputs {
                    let nested = i.make(offset + text.len(), Some(i.hir_id), scx)?;
                    text.push_str(&nested.text);
                    text.push(',');
                    defs.extend(nested.defs.into_iter());
                    refs.extend(nested.refs.into_iter());
                }
                text.push(')');
                if let hir::FnRetTy::Return(ref t) = f.decl.output {
                    text.push_str(" -> ");
                    let nested = t.make(offset + text.len(), None, scx)?;
                    text.push_str(&nested.text);
                    text.push(',');
                    defs.extend(nested.defs.into_iter());
                    refs.extend(nested.refs.into_iter());
                }

                Ok(Signature { text, defs, refs })
            }
            hir::TyKind::Path(hir::QPath::Resolved(None, ref path)) => path.make(offset, id, scx),
            hir::TyKind::Path(hir::QPath::Resolved(Some(ref qself), ref path)) => {
                let nested_ty = qself.make(offset + 1, id, scx)?;
                let prefix = format!(
                    "<{} as {}>::",
                    nested_ty.text,
                    path_segment_to_string(&path.segments[0])
                );

                let name = path_segment_to_string(path.segments.last().ok_or("Bad path")?);
                let res = scx.get_path_res(id.ok_or("Missing id for Path")?);
                let id = id_from_def_id(res.def_id());
                if path.segments.len() == 2 {
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
                    // FIXME should put the proper path in there, not ellipsis.
                    Ok(Signature {
                        text: prefix + "...::" + &name,
                        defs: vec![],
                        refs: vec![SigElement { id, start, end }],
                    })
                }
            }
            hir::TyKind::Path(hir::QPath::TypeRelative(ty, segment)) => {
                let nested_ty = ty.make(offset + 1, id, scx)?;
                let prefix = format!("<{}>::", nested_ty.text);

                let name = path_segment_to_string(segment);
                let res = scx.get_path_res(id.ok_or("Missing id for Path")?);
                let id = id_from_def_id(res.def_id());

                let start = offset + prefix.len();
                let end = start + name.len();
                Ok(Signature {
                    text: prefix + &name,
                    defs: vec![],
                    refs: vec![SigElement { id, start, end }],
                })
            }
            hir::TyKind::Path(hir::QPath::LangItem(lang_item, _, _)) => {
                Ok(text_sig(format!("#[lang = \"{}\"]", lang_item.name())))
            }
            hir::TyKind::TraitObject(bounds, ..) => {
                // FIXME recurse into bounds
                let bounds: Vec<hir::GenericBound<'_>> = bounds
                    .iter()
                    .map(|hir::PolyTraitRef { bound_generic_params, trait_ref, span }| {
                        hir::GenericBound::Trait(
                            hir::PolyTraitRef {
                                bound_generic_params,
                                trait_ref: hir::TraitRef {
                                    path: trait_ref.path,
                                    hir_ref_id: trait_ref.hir_ref_id,
                                },
                                span: *span,
                            },
                            hir::TraitBoundModifier::None,
                        )
                    })
                    .collect();
                let nested = bounds_to_string(&bounds);
                Ok(text_sig(nested))
            }
            hir::TyKind::Array(ref ty, ref length) => {
                let nested_ty = ty.make(offset + 1, id, scx)?;
                let expr = id_to_string(&scx.tcx.hir(), length.hir_id()).replace('\n', " ");
                let text = format!("[{}; {}]", nested_ty.text, expr);
                Ok(replace_text(nested_ty, text))
            }
            hir::TyKind::OpaqueDef(item_id, _, _) => {
                let item = scx.tcx.hir().item(item_id);
                item.make(offset, Some(item_id.hir_id()), scx)
            }
            hir::TyKind::Typeof(_) | hir::TyKind::Infer | hir::TyKind::Err => Err("Ty"),
        }
    }
}

impl<'hir> Sig for hir::Item<'hir> {
    type Parent = hir::HirId;
    fn make(&self, offset: usize, _parent_id: Option<hir::HirId>, scx: &SaveContext<'_>) -> Result {
        let id = Some(self.hir_id());

        match self.kind {
            hir::ItemKind::Static(ref ty, m, ref body) => {
                let mut text = "static ".to_owned();
                if m.is_mut() {
                    text.push_str("mut ");
                }
                let name = self.ident.to_string();
                let defs = vec![SigElement {
                    id: id_from_def_id(self.owner_id.to_def_id()),
                    start: offset + text.len(),
                    end: offset + text.len() + name.len(),
                }];
                text.push_str(&name);
                text.push_str(": ");

                let ty = ty.make(offset + text.len(), id, scx)?;
                text.push_str(&ty.text);

                text.push_str(" = ");
                let expr = id_to_string(&scx.tcx.hir(), body.hir_id).replace('\n', " ");
                text.push_str(&expr);

                text.push(';');

                Ok(extend_sig(ty, text, defs, vec![]))
            }
            hir::ItemKind::Const(ref ty, ref body) => {
                let mut text = "const ".to_owned();
                let name = self.ident.to_string();
                let defs = vec![SigElement {
                    id: id_from_def_id(self.owner_id.to_def_id()),
                    start: offset + text.len(),
                    end: offset + text.len() + name.len(),
                }];
                text.push_str(&name);
                text.push_str(": ");

                let ty = ty.make(offset + text.len(), id, scx)?;
                text.push_str(&ty.text);

                text.push_str(" = ");
                let expr = id_to_string(&scx.tcx.hir(), body.hir_id).replace('\n', " ");
                text.push_str(&expr);

                text.push(';');

                Ok(extend_sig(ty, text, defs, vec![]))
            }
            hir::ItemKind::Fn(hir::FnSig { ref decl, header, span: _ }, ref generics, _) => {
                let mut text = String::new();
                if let hir::Constness::Const = header.constness {
                    text.push_str("const ");
                }
                if hir::IsAsync::Async == header.asyncness {
                    text.push_str("async ");
                }
                if let hir::Unsafety::Unsafe = header.unsafety {
                    text.push_str("unsafe ");
                }
                text.push_str("fn ");

                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;

                sig.text.push('(');
                for i in decl.inputs {
                    // FIXME should descend into patterns to add defs.
                    sig.text.push_str(": ");
                    let nested = i.make(offset + sig.text.len(), Some(i.hir_id), scx)?;
                    sig.text.push_str(&nested.text);
                    sig.text.push(',');
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push(')');

                if let hir::FnRetTy::Return(ref t) = decl.output {
                    sig.text.push_str(" -> ");
                    let nested = t.make(offset + sig.text.len(), None, scx)?;
                    sig.text.push_str(&nested.text);
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push_str(" {}");

                Ok(sig)
            }
            hir::ItemKind::Macro(..) => {
                let mut text = "macro".to_owned();
                let name = self.ident.to_string();
                text.push_str(&name);
                text.push_str(&"! {}");

                Ok(text_sig(text))
            }
            hir::ItemKind::Mod(ref _mod) => {
                let mut text = "mod ".to_owned();
                let name = self.ident.to_string();
                let defs = vec![SigElement {
                    id: id_from_def_id(self.owner_id.to_def_id()),
                    start: offset + text.len(),
                    end: offset + text.len() + name.len(),
                }];
                text.push_str(&name);
                // Could be either `mod foo;` or `mod foo { ... }`, but we'll just pick one.
                text.push(';');

                Ok(Signature { text, defs, refs: vec![] })
            }
            hir::ItemKind::TyAlias(ref ty, ref generics) => {
                let text = "type ".to_owned();
                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;

                sig.text.push_str(" = ");
                let ty = ty.make(offset + sig.text.len(), id, scx)?;
                sig.text.push_str(&ty.text);
                sig.text.push(';');

                Ok(merge_sigs(sig.text.clone(), vec![sig, ty]))
            }
            hir::ItemKind::Enum(_, ref generics) => {
                let text = "enum ".to_owned();
                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;
                sig.text.push_str(" {}");
                Ok(sig)
            }
            hir::ItemKind::Struct(_, ref generics) => {
                let text = "struct ".to_owned();
                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;
                sig.text.push_str(" {}");
                Ok(sig)
            }
            hir::ItemKind::Union(_, ref generics) => {
                let text = "union ".to_owned();
                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;
                sig.text.push_str(" {}");
                Ok(sig)
            }
            hir::ItemKind::Trait(is_auto, unsafety, ref generics, bounds, _) => {
                let mut text = String::new();

                if is_auto == hir::IsAuto::Yes {
                    text.push_str("auto ");
                }

                if let hir::Unsafety::Unsafe = unsafety {
                    text.push_str("unsafe ");
                }
                text.push_str("trait ");
                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;

                if !bounds.is_empty() {
                    sig.text.push_str(": ");
                    sig.text.push_str(&bounds_to_string(bounds));
                }
                // FIXME where clause
                sig.text.push_str(" {}");

                Ok(sig)
            }
            hir::ItemKind::TraitAlias(ref generics, bounds) => {
                let mut text = String::new();
                text.push_str("trait ");
                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;

                if !bounds.is_empty() {
                    sig.text.push_str(" = ");
                    sig.text.push_str(&bounds_to_string(bounds));
                }
                // FIXME where clause
                sig.text.push(';');

                Ok(sig)
            }
            hir::ItemKind::Impl(hir::Impl {
                unsafety,
                polarity,
                defaultness,
                defaultness_span: _,
                constness,
                ref generics,
                ref of_trait,
                ref self_ty,
                items: _,
            }) => {
                let mut text = String::new();
                if let hir::Defaultness::Default { .. } = defaultness {
                    text.push_str("default ");
                }
                if let hir::Unsafety::Unsafe = unsafety {
                    text.push_str("unsafe ");
                }
                text.push_str("impl");
                if let hir::Constness::Const = constness {
                    text.push_str(" const");
                }

                let generics_sig =
                    generics.make(offset + text.len(), Some(self.owner_id.def_id), scx)?;
                text.push_str(&generics_sig.text);

                text.push(' ');

                let trait_sig = if let Some(ref t) = *of_trait {
                    if let hir::ImplPolarity::Negative(_) = polarity {
                        text.push('!');
                    }
                    let trait_sig = t.path.make(offset + text.len(), id, scx)?;
                    text.push_str(&trait_sig.text);
                    text.push_str(" for ");
                    trait_sig
                } else {
                    text_sig(String::new())
                };

                let ty_sig = self_ty.make(offset + text.len(), id, scx)?;
                text.push_str(&ty_sig.text);

                text.push_str(" {}");

                Ok(merge_sigs(text, vec![generics_sig, trait_sig, ty_sig]))

                // FIXME where clause
            }
            hir::ItemKind::ForeignMod { .. } => Err("extern mod"),
            hir::ItemKind::GlobalAsm(_) => Err("global asm"),
            hir::ItemKind::ExternCrate(_) => Err("extern crate"),
            hir::ItemKind::OpaqueTy(ref opaque) => {
                if opaque.in_trait {
                    Err("opaque type in trait")
                } else {
                    Err("opaque type")
                }
            }
            // FIXME should implement this (e.g., pub use).
            hir::ItemKind::Use(..) => Err("import"),
        }
    }
}

impl<'hir> Sig for hir::Path<'hir> {
    type Parent = hir::HirId;
    fn make(&self, offset: usize, id: Option<hir::HirId>, scx: &SaveContext<'_>) -> Result {
        let res = scx.get_path_res(id.ok_or("Missing id for Path")?);

        let (name, start, end) = match res {
            Res::PrimTy(..) | Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } | Res::Err => {
                return Ok(Signature { text: path_to_string(self), defs: vec![], refs: vec![] });
            }
            Res::Def(DefKind::AssocConst | DefKind::Variant | DefKind::Ctor(..), _) => {
                let len = self.segments.len();
                if len < 2 {
                    return Err("Bad path");
                }
                // FIXME: really we should descend into the generics here and add SigElements for
                // them.
                // FIXME: would be nice to have a def for the first path segment.
                let seg1 = path_segment_to_string(&self.segments[len - 2]);
                let seg2 = path_segment_to_string(&self.segments[len - 1]);
                let start = offset + seg1.len() + 2;
                (format!("{}::{}", seg1, seg2), start, start + seg2.len())
            }
            _ => {
                let name = path_segment_to_string(self.segments.last().ok_or("Bad path")?);
                let end = offset + name.len();
                (name, offset, end)
            }
        };

        let id = id_from_def_id(res.def_id());
        Ok(Signature { text: name, defs: vec![], refs: vec![SigElement { id, start, end }] })
    }
}

// This does not cover the where clause, which must be processed separately.
impl<'hir> Sig for hir::Generics<'hir> {
    type Parent = LocalDefId;
    fn make(&self, offset: usize, _parent_id: Option<LocalDefId>, scx: &SaveContext<'_>) -> Result {
        if self.params.is_empty() {
            return Ok(text_sig(String::new()));
        }

        let mut text = "<".to_owned();

        let mut defs = Vec::with_capacity(self.params.len());
        for param in self.params {
            let mut param_text = String::new();
            if let hir::GenericParamKind::Const { .. } = param.kind {
                param_text.push_str("const ");
            }
            param_text.push_str(param.name.ident().as_str());
            defs.push(SigElement {
                id: id_from_def_id(param.def_id.to_def_id()),
                start: offset + text.len(),
                end: offset + text.len() + param_text.as_str().len(),
            });
            if let hir::GenericParamKind::Const { ref ty, default } = param.kind {
                param_text.push_str(": ");
                param_text.push_str(&ty_to_string(&ty));
                if let Some(default) = default {
                    param_text.push_str(" = ");
                    param_text.push_str(&id_to_string(&scx.tcx.hir(), default.hir_id));
                }
            }
            text.push_str(&param_text);
            text.push(',');
        }

        text.push('>');
        Ok(Signature { text, defs, refs: vec![] })
    }
}

impl<'hir> Sig for hir::FieldDef<'hir> {
    type Parent = LocalDefId;
    fn make(&self, offset: usize, _parent_id: Option<LocalDefId>, scx: &SaveContext<'_>) -> Result {
        let mut text = String::new();

        text.push_str(&self.ident.to_string());
        let defs = Some(SigElement {
            id: id_from_def_id(self.def_id.to_def_id()),
            start: offset,
            end: offset + text.len(),
        });
        text.push_str(": ");

        let mut ty_sig = self.ty.make(offset + text.len(), Some(self.hir_id), scx)?;
        text.push_str(&ty_sig.text);
        ty_sig.text = text;
        ty_sig.defs.extend(defs.into_iter());
        Ok(ty_sig)
    }
}

impl<'hir> Sig for hir::Variant<'hir> {
    type Parent = LocalDefId;
    fn make(&self, offset: usize, parent_id: Option<LocalDefId>, scx: &SaveContext<'_>) -> Result {
        let mut text = self.ident.to_string();
        match self.data {
            hir::VariantData::Struct(fields, r) => {
                let id = parent_id.ok_or("Missing id for Variant's parent")?;
                let name_def = SigElement {
                    id: id_from_def_id(id.to_def_id()),
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
            hir::VariantData::Tuple(fields, _, def_id) => {
                let name_def = SigElement {
                    id: id_from_def_id(def_id.to_def_id()),
                    start: offset,
                    end: offset + text.len(),
                };
                text.push('(');
                let mut defs = vec![name_def];
                let mut refs = vec![];
                for f in fields {
                    let field_sig = f.make(offset + text.len(), Some(def_id), scx)?;
                    text.push_str(&field_sig.text);
                    text.push_str(", ");
                    defs.extend(field_sig.defs.into_iter());
                    refs.extend(field_sig.refs.into_iter());
                }
                text.push(')');
                Ok(Signature { text, defs, refs })
            }
            hir::VariantData::Unit(_, def_id) => {
                let name_def = SigElement {
                    id: id_from_def_id(def_id.to_def_id()),
                    start: offset,
                    end: offset + text.len(),
                };
                Ok(Signature { text, defs: vec![name_def], refs: vec![] })
            }
        }
    }
}

impl<'hir> Sig for hir::ForeignItem<'hir> {
    type Parent = hir::HirId;
    fn make(&self, offset: usize, _parent_id: Option<hir::HirId>, scx: &SaveContext<'_>) -> Result {
        let id = Some(self.hir_id());
        match self.kind {
            hir::ForeignItemKind::Fn(decl, _, ref generics) => {
                let mut text = String::new();
                text.push_str("fn ");

                let mut sig =
                    name_and_generics(text, offset, generics, self.owner_id, self.ident, scx)?;

                sig.text.push('(');
                for i in decl.inputs {
                    sig.text.push_str(": ");
                    let nested = i.make(offset + sig.text.len(), Some(i.hir_id), scx)?;
                    sig.text.push_str(&nested.text);
                    sig.text.push(',');
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push(')');

                if let hir::FnRetTy::Return(ref t) = decl.output {
                    sig.text.push_str(" -> ");
                    let nested = t.make(offset + sig.text.len(), None, scx)?;
                    sig.text.push_str(&nested.text);
                    sig.defs.extend(nested.defs.into_iter());
                    sig.refs.extend(nested.refs.into_iter());
                }
                sig.text.push(';');

                Ok(sig)
            }
            hir::ForeignItemKind::Static(ref ty, m) => {
                let mut text = "static ".to_owned();
                if m == Mutability::Mut {
                    text.push_str("mut ");
                }
                let name = self.ident.to_string();
                let defs = vec![SigElement {
                    id: id_from_def_id(self.owner_id.to_def_id()),
                    start: offset + text.len(),
                    end: offset + text.len() + name.len(),
                }];
                text.push_str(&name);
                text.push_str(": ");

                let ty_sig = ty.make(offset + text.len(), id, scx)?;
                text.push(';');

                Ok(extend_sig(ty_sig, text, defs, vec![]))
            }
            hir::ForeignItemKind::Type => {
                let mut text = "type ".to_owned();
                let name = self.ident.to_string();
                let defs = vec![SigElement {
                    id: id_from_def_id(self.owner_id.to_def_id()),
                    start: offset + text.len(),
                    end: offset + text.len() + name.len(),
                }];
                text.push_str(&name);
                text.push(';');

                Ok(Signature { text, defs, refs: vec![] })
            }
        }
    }
}

fn name_and_generics(
    mut text: String,
    offset: usize,
    generics: &hir::Generics<'_>,
    id: hir::OwnerId,
    name: Ident,
    scx: &SaveContext<'_>,
) -> Result {
    let name = name.to_string();
    let def = SigElement {
        id: id_from_def_id(id.to_def_id()),
        start: offset + text.len(),
        end: offset + text.len() + name.len(),
    };
    text.push_str(&name);
    let generics: Signature = generics.make(offset + text.len(), Some(id.def_id), scx)?;
    // FIXME where clause
    let text = format!("{}{}", text, generics.text);
    Ok(extend_sig(generics, text, vec![def], vec![]))
}

fn make_assoc_type_signature(
    id: hir::OwnerId,
    ident: Ident,
    bounds: Option<hir::GenericBounds<'_>>,
    default: Option<&hir::Ty<'_>>,
    scx: &SaveContext<'_>,
) -> Result {
    let mut text = "type ".to_owned();
    let name = ident.to_string();
    let mut defs = vec![SigElement {
        id: id_from_def_id(id.to_def_id()),
        start: text.len(),
        end: text.len() + name.len(),
    }];
    let mut refs = vec![];
    text.push_str(&name);
    if let Some(bounds) = bounds {
        text.push_str(": ");
        // FIXME should descend into bounds
        text.push_str(&bounds_to_string(bounds));
    }
    if let Some(default) = default {
        text.push_str(" = ");
        let ty_sig = default.make(text.len(), Some(id.into()), scx)?;
        text.push_str(&ty_sig.text);
        defs.extend(ty_sig.defs.into_iter());
        refs.extend(ty_sig.refs.into_iter());
    }
    text.push(';');
    Ok(Signature { text, defs, refs })
}

fn make_assoc_const_signature(
    id: hir::OwnerId,
    ident: Symbol,
    ty: &hir::Ty<'_>,
    default: Option<&hir::Expr<'_>>,
    scx: &SaveContext<'_>,
) -> Result {
    let mut text = "const ".to_owned();
    let name = ident.to_string();
    let mut defs = vec![SigElement {
        id: id_from_def_id(id.to_def_id()),
        start: text.len(),
        end: text.len() + name.len(),
    }];
    let mut refs = vec![];
    text.push_str(&name);
    text.push_str(": ");

    let ty_sig = ty.make(text.len(), Some(id.into()), scx)?;
    text.push_str(&ty_sig.text);
    defs.extend(ty_sig.defs.into_iter());
    refs.extend(ty_sig.refs.into_iter());

    if let Some(default) = default {
        text.push_str(" = ");
        text.push_str(&id_to_string(&scx.tcx.hir(), default.hir_id));
    }
    text.push(';');
    Ok(Signature { text, defs, refs })
}

fn make_method_signature(
    id: hir::OwnerId,
    ident: Ident,
    generics: &hir::Generics<'_>,
    m: &hir::FnSig<'_>,
    scx: &SaveContext<'_>,
) -> Result {
    // FIXME code dup with function signature
    let mut text = String::new();
    if let hir::Constness::Const = m.header.constness {
        text.push_str("const ");
    }
    if hir::IsAsync::Async == m.header.asyncness {
        text.push_str("async ");
    }
    if let hir::Unsafety::Unsafe = m.header.unsafety {
        text.push_str("unsafe ");
    }
    text.push_str("fn ");

    let mut sig = name_and_generics(text, 0, generics, id, ident, scx)?;

    sig.text.push('(');
    for i in m.decl.inputs {
        sig.text.push_str(": ");
        let nested = i.make(sig.text.len(), Some(i.hir_id), scx)?;
        sig.text.push_str(&nested.text);
        sig.text.push(',');
        sig.defs.extend(nested.defs.into_iter());
        sig.refs.extend(nested.refs.into_iter());
    }
    sig.text.push(')');

    if let hir::FnRetTy::Return(ref t) = m.decl.output {
        sig.text.push_str(" -> ");
        let nested = t.make(sig.text.len(), None, scx)?;
        sig.text.push_str(&nested.text);
        sig.defs.extend(nested.defs.into_iter());
        sig.refs.extend(nested.refs.into_iter());
    }
    sig.text.push_str(" {}");

    Ok(sig)
}
