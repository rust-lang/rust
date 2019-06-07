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

use rustc::hir::def::{DefKind, Res};
use syntax::ast::{self, NodeId};
use syntax::print::pprust;

use std::fmt::Write;

use log::warn;

pub fn item_signature(item: &ast::Item, scx: &SaveContext<'_, '_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    let mut b = SigBuilder::new();
    item.make(&mut b, None, scx).map_err(|e| warn!("item sig failed: {}", e)).ok()?;
    Some(b.build())
}

pub fn foreign_item_signature(
    item: &ast::ForeignItem,
    scx: &SaveContext<'_, '_>,
) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    let mut b = SigBuilder::new();
    item.make(&mut b, None, scx).map_err(|e| warn!("item sig failed: {}", e)).ok();
    Some(b.build())
}

/// Signature for a struct or tuple field declaration.
/// Does not include a trailing comma.
pub fn field_signature(field: &ast::StructField, scx: &SaveContext<'_, '_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    let mut b = SigBuilder::new();
    field.make(&mut b, None, scx).map_err(|e| warn!("item sig failed: {}", e)).ok()?;
    Some(b.build())
}

/// Does not include a trailing comma.
pub fn variant_signature(variant: &ast::Variant, scx: &SaveContext<'_, '_>) -> Option<Signature> {
    if !scx.config.signatures {
        return None;
    }
    let mut b = SigBuilder::new();
    variant.node.make(&mut b, None, scx).map_err(|e| warn!("item sig failed: {}", e)).ok()?;
    Some(b.build())
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
    let mut b = SigBuilder::new();
    make_method_signature(&mut b, id, ident, generics, m, scx)
        .map_err(|e| warn!("item sig failed: {}", e))
        .ok()?;
    Some(b.build())
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

    let mut b = SigBuilder::new();
    make_assoc_const_signature(&mut b, id, ident, ty, default, scx)
        .map_err(|e| warn!("item sig failed: {}", e))
        .ok()?;
    Some(b.build())
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

    let mut b = SigBuilder::new();
    make_assoc_type_signature(&mut b, id, ident, bounds, default, scx)
        .map_err(|e| warn!("item sig failed: {}", e))
        .ok()?;
    Some(b.build())
}

struct SigBuilder {
    text: String,
    refs: Vec<SigElement>,
    defs: Vec<SigElement>,
}

impl SigBuilder {
    fn new() -> SigBuilder {
        SigBuilder { text: String::new(), refs: Vec::new(), defs: Vec::new() }
    }
    fn build(self) -> Signature {
        let SigBuilder { refs, defs, text } = self;
        Signature { refs, defs, text }
    }
}
macro_rules! text {
    ($builder:expr, $($rest:expr),+) => ({
        // note: writing to a string can't fail
        let _ = write!(&mut $builder.text, $($rest),+);
    });
}
macro_rules! ref_ {
    ($builder:expr, $id:expr, $($rest:expr),+) => ({
        let start = $builder.text.len();
        let _ = write!(&mut $builder.text, $($rest),+);
        let end = $builder.text.len();
        $builder.refs.push(SigElement { start, end, id: $id });
    });
}
macro_rules! def_ {
    ($builder:expr, $id:expr, $($rest:expr),+) => ({
        let start = $builder.text.len();
        let _ = write!(&mut $builder.text, $($rest),+);
        let end = $builder.text.len();
        $builder.defs.push(SigElement { start, end, id: $id });
    });
}

type Result = std::result::Result<(), &'static str>;

trait Sig {
    fn make(
        &self,
        builder: &mut SigBuilder,
        parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result;
}

impl Sig for ast::Ty {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        let id = Some(self.id);
        match self.node {
            ast::TyKind::Slice(ref ty) => {
                text!(b, "[");
                ty.make(b, id, scx)?;
                text!(b, "]");
            }
            ast::TyKind::Ptr(ref mt) => {
                match mt.mutbl {
                    ast::Mutability::Mutable => text!(b, "*mut "),
                    ast::Mutability::Immutable => text!(b, "*const "),
                }
                mt.ty.make(b, id, scx)?;
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                text!(b, "&");
                if let &Some(ref l) = lifetime {
                    // FIXME id?
                    text!(b, "{} ", l.ident);
                }
                if let ast::Mutability::Mutable = mt.mutbl {
                    text!(b, "mut ");
                };
                mt.ty.make(b, id, scx)?;
            }
            ast::TyKind::Never => text!(b, "!"),
            ast::TyKind::CVarArgs => text!(b, "..."),
            ast::TyKind::Tup(ref ts) => {
                text!(b, "(");
                for t in ts {
                    t.make(b, id, scx)?;
                    text!(b, ",");
                }
                text!(b, ")")
            }
            ast::TyKind::Paren(ref ty) => {
                text!(b, "(");
                ty.make(b, id, scx)?;
                text!(b, ")");
            }
            ast::TyKind::BareFn(ref f) => {
                if !f.generic_params.is_empty() {
                    // FIXME defs, bounds on lifetimes
                    let content = &f
                        .generic_params
                        .iter()
                        .filter_map(|param| match param.kind {
                            ast::GenericParamKind::Lifetime { .. } => Some(param.ident.to_string()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(", ");

                    text!(b, "for<{}> ", content);
                }

                if f.unsafety == ast::Unsafety::Unsafe {
                    text!(b, "unsafe ");
                }
                if f.abi != rustc_target::spec::abi::Abi::Rust {
                    text!(b, "extern {} ", f.abi.to_string());
                }
                text!(b, "fn");

                for i in &f.decl.inputs {
                    i.ty.make(b, Some(i.id), scx)?;
                    text!(b, ",");
                }
                text!(b, ")");
                if let ast::FunctionRetTy::Ty(ref t) = f.decl.output {
                    text!(b, " -> ");
                    t.make(b, None, scx)?;
                    // FIXME should this comma be here??
                    text!(b, ",");
                }
            }
            ast::TyKind::Path(None, ref path) => path.make(b, id, scx)?,
            ast::TyKind::Path(Some(ref qself), ref path) => {
                text!(b, "<");
                qself.ty.make(b, id, scx)?;
                if qself.position == 0 {
                    text!(b, ">::");
                } else if qself.position == 1 {
                    // FIXME ref
                    let first = pprust::path_segment_to_string(&path.segments[0]);
                    text!(b, " as {}>::", first);
                } else {
                    // FIXME proper path
                    text!(b, " as ...>::");
                }

                let name = pprust::path_segment_to_string(path.segments.last().ok_or("Bad path")?);
                let res = scx.get_path_res(id.ok_or("Missing id for Path")?);
                let id = id_from_def_id(res.def_id());
                if path.segments.len() - qself.position == 1 {
                    ref_!(b, id, "{}", name);
                } else {
                    // FIXME should put the proper path in there, not elipses.
                    ref_!(b, id, "...::{}", name);
                }
            }
            ast::TyKind::TraitObject(ref bounds, ..) => {
                // FIXME recurse into bounds
                let nested = pprust::bounds_to_string(bounds);
                text!(b, "{}", nested);
            }
            ast::TyKind::ImplTrait(_, ref bounds) => {
                // FIXME recurse into bounds
                let nested = pprust::bounds_to_string(bounds);
                text!(b, "impl {}", nested);
            }
            ast::TyKind::Array(ref ty, ref v) => {
                text!(b, "[");
                ty.make(b, id, scx)?;
                let expr = pprust::expr_to_string(&v.value).replace('\n', " ");
                text!(b, "; {}]", expr);
            }
            ast::TyKind::Typeof(_)
            | ast::TyKind::Infer
            | ast::TyKind::Err
            | ast::TyKind::ImplicitSelf
            | ast::TyKind::Mac(_) => Err("Ty")?,
        }
        Ok(())
    }
}

impl Sig for ast::Item {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        let id = Some(self.id);

        match self.node {
            ast::ItemKind::Static(ref ty, m, ref expr) => {
                text!(b, "static ");
                if m == ast::Mutability::Mutable {
                    text!(b, "mut ");
                }
                def_!(b, id_from_node_id(self.id, scx), "{}", self.ident.to_string());
                text!(b, ": ");
                ty.make(b, id, scx)?;

                let expr = pprust::expr_to_string(expr).replace('\n', " ");

                text!(b, " = {};", expr);
            }
            ast::ItemKind::Const(ref ty, ref expr) => {
                text!(b, "const ");
                def_!(b, id_from_node_id(self.id, scx), "{}", self.ident.to_string());
                text!(b, ": ");
                ty.make(b, id, scx)?;

                let expr = pprust::expr_to_string(expr).replace('\n', " ");

                text!(b, " = {};", expr);
            }
            ast::ItemKind::Fn(ref decl, header, ref generics, _) => {
                header.make(b, None, scx)?;
                name_and_generics(b, generics, self.id, self.ident, scx)?;
                text!(b, "(");

                for i in &decl.inputs {
                    // FIXME should descend into patterns to add defs.
                    text!(b, "{}: ", &pprust::pat_to_string(&i.pat));
                    i.ty.make(b, Some(i.id), scx)?;
                    // FIXME trailing comma
                    text!(b, ", ");
                }
                text!(b, ")");

                if let ast::FunctionRetTy::Ty(ref t) = decl.output {
                    text!(b, " -> ");
                    t.make(b, None, scx)?;
                }
                text!(b, " {{}}"); // escaping for write!
            }
            ast::ItemKind::Mod(ref _mod) => {
                text!(b, "mod ");
                let id = id_from_node_id(self.id, scx);
                def_!(b, id, "{}", self.ident.to_string());
                // Could be either `mod foo;` or `mod foo { ... }`, but we'll just pick one.
                text!(b, ";");
            }
            ast::ItemKind::Existential(ref bounds, ref generics) => {
                text!(b, "existential type ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;

                if !bounds.is_empty() {
                    // FIXME descend
                    text!(b, ": {}", pprust::bounds_to_string(bounds));
                }
                text!(b, ";");
            }
            ast::ItemKind::Ty(ref ty, ref generics) => {
                text!(b, "type ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;

                text!(b, " = ");
                ty.make(b, id, scx)?;
                text!(b, ";");
            }
            // FIXME enum variants / struct fields?
            ast::ItemKind::Enum(_, ref generics) => {
                text!(b, "enum ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;
                text!(b, " {{}}");
            }
            ast::ItemKind::Struct(_, ref generics) => {
                text!(b, "struct ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;
                text!(b, " {{}}");
            }
            ast::ItemKind::Union(_, ref generics) => {
                text!(b, "union ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;
                text!(b, " {{}}");
            }
            ast::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, _) => {
                if is_auto == ast::IsAuto::Yes {
                    text!(b, "auto ");
                }

                if unsafety == ast::Unsafety::Unsafe {
                    text!(b, "unsafe ");
                }
                text!(b, "trait ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;

                if !bounds.is_empty() {
                    // FIXME descend into bounds
                    text!(b, ": {}", pprust::bounds_to_string(bounds));
                }
                // FIXME where clause
                text!(b, " {{}}");
            }
            ast::ItemKind::TraitAlias(ref generics, ref bounds) => {
                text!(b, "trait ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;

                if !bounds.is_empty() {
                    text!(b, " = {}", pprust::bounds_to_string(bounds));
                }
                // FIXME where clause
                text!(b, ";");
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
                if let ast::Defaultness::Default = defaultness {
                    text!(b, "default ");
                }
                if unsafety == ast::Unsafety::Unsafe {
                    text!(b, "unsafe ");
                }
                text!(b, "impl ");

                generics.make(b, id, scx)?;

                text!(b, " ");

                if let Some(ref t) = *opt_trait {
                    if polarity == ast::ImplPolarity::Negative {
                        text!(b, "!");
                    }
                    t.path.make(b, id, scx)?;
                    text!(b, " for ");
                }

                ty.make(b, id, scx)?;
                text!(b, " {{}}");

                // FIXME where clause
            }
            ast::ItemKind::ForeignMod(_) => Err("extern mod")?,
            ast::ItemKind::GlobalAsm(_) => Err("glboal asm")?,
            ast::ItemKind::ExternCrate(_) => Err("extern crate")?,
            // FIXME should implement this (e.g., pub use).
            ast::ItemKind::Use(_) => Err("import")?,
            ast::ItemKind::Mac(..) | ast::ItemKind::MacroDef(_) => Err("Macro")?,
        }
        Ok(())
    }
}

// `const async unsafe extern "abi" fn`
impl Sig for ast::FnHeader {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        _scx: &SaveContext<'_, '_>,
    ) -> Result {
        if self.constness.node == ast::Constness::Const {
            text!(b, "const ");
        }
        if self.asyncness.node.is_async() {
            text!(b, "async ");
        }
        if self.unsafety == ast::Unsafety::Unsafe {
            text!(b, "unsafe ");
        }
        if self.abi != rustc_target::spec::abi::Abi::Rust {
            text!(b, "extern {} ", self.abi.to_string());
        }
        text!(b, "fn ");

        Ok(())
    }
}

impl Sig for ast::Path {
    fn make(
        &self,
        b: &mut SigBuilder,
        parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        let res = scx.get_path_res(parent_id.ok_or("Missing id for Path")?);

        match res {
            Res::PrimTy(..) | Res::SelfTy(..) | Res::Err => {
                text!(b, "{}", pprust::path_to_string(self));
            }
            Res::Def(DefKind::AssocConst, _)
            | Res::Def(DefKind::Variant, _)
            | Res::Def(DefKind::Ctor(..), _) => {
                let len = self.segments.len();
                if len < 2 {
                    // FIXME can we accept longer paths here?
                    return Err("Bad path");
                }
                // FIXME: really we should descend into the generics here and add SigElements for
                // them.
                // FIXME: would be nice to have a def for the first path segment.
                let seg1 = pprust::path_segment_to_string(&self.segments[len - 2]);
                let seg2 = pprust::path_segment_to_string(&self.segments[len - 1]);
                ref_!(b, id_from_def_id(res.def_id()), "{}::{}", seg1, seg2);
            }
            _ => {
                let name = pprust::path_segment_to_string(self.segments.last().ok_or("Bad path")?);
                ref_!(b, id_from_def_id(res.def_id()), "{}", name);
            }
        }

        Ok(())
    }
}

// This does not cover the where clause, which must be processed separately.
impl Sig for ast::Generics {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        if self.params.is_empty() {
            return Ok(());
        }

        text!(b, "<");

        for param in &self.params {
            if let ast::GenericParamKind::Const { .. } = param.kind {
                text!(b, "const ");
            }
            def_!(b, id_from_node_id(param.id, scx), "{}", param.ident.as_str());
            if let ast::GenericParamKind::Const { ref ty } = param.kind {
                // FIXME descend
                text!(b, ": {}", pprust::ty_to_string(&ty));
            }
            if !param.bounds.is_empty() {
                text!(b, ": ");
                match param.kind {
                    ast::GenericParamKind::Lifetime { .. } => {
                        let bounds = param
                            .bounds
                            .iter()
                            .map(|bound| match bound {
                                ast::GenericBound::Outlives(lt) => lt.ident.to_string(),
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>()
                            .join(" + ");
                        text!(b, "{}", &bounds);
                        // FIXME add lifetime bounds refs.
                    }
                    ast::GenericParamKind::Type { .. } => {
                        text!(b, "{}", pprust::bounds_to_string(&param.bounds));
                        // FIXME descend properly into bounds.
                    }
                    ast::GenericParamKind::Const { .. } => {
                        // Const generics cannot contain bounds.
                    }
                }
            }
            // FIXME trailing comma
            text!(b, ",");
        }
        text!(b, ">");
        Ok(())
    }
}

impl Sig for ast::StructField {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        if let Some(ident) = self.ident {
            def_!(b, id_from_node_id(self.id, scx), "{}", ident.to_string());
            text!(b, ": ");
        }

        self.ty.make(b, Some(self.id), scx)?;
        Ok(())
    }
}

impl Sig for ast::Variant_ {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        let id = self.id;
        match self.data {
            ast::VariantData::Struct(ref fields, r) => {
                def_!(b, id_from_node_id(id, scx), "{}", self.ident.to_string());
                text!(b, " {{ ");
                if r {
                    // FIXME parse error?
                    text!(b, "/* parse error */ ");
                } else {
                    for f in fields {
                        f.make(b, Some(id), scx)?;
                        text!(b, ", ");
                    }
                }
                text!(b, "}}");
            }
            ast::VariantData::Tuple(ref fields, id) => {
                def_!(b, id_from_node_id(id, scx), "{}", self.ident.to_string());
                text!(b, "(");
                for f in fields {
                    f.make(b, Some(id), scx)?;
                    text!(b, ", ");
                }
                text!(b, ")");
            }
            ast::VariantData::Unit(id) => {
                def_!(b, id_from_node_id(id, scx), "{}", self.ident.to_string());
            }
        }
        Ok(())
    }
}

impl Sig for ast::ForeignItem {
    fn make(
        &self,
        b: &mut SigBuilder,
        _parent_id: Option<NodeId>,
        scx: &SaveContext<'_, '_>,
    ) -> Result {
        let id = Some(self.id);
        match self.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                text!(b, "fn ");
                name_and_generics(b, generics, self.id, self.ident, scx)?;
                text!(b, "(");
                for i in &decl.inputs {
                    // FIXME should descend into patterns to add defs.
                    text!(b, "{}: ", pprust::pat_to_string(&i.pat));
                    i.ty.make(b, Some(i.id), scx)?;
                    // FIXME trailing comma
                    text!(b, ",");
                }
                text!(b, ")");

                if let ast::FunctionRetTy::Ty(ref t) = decl.output {
                    text!(b, " -> ");
                    t.make(b, None, scx)?;
                }
                // FIXME where clause
                text!(b, ";");
            }
            ast::ForeignItemKind::Static(ref ty, m) => {
                text!(b, "static ");
                if m == ast::Mutability::Mutable {
                    text!(b, "mut ");
                }
                def_!(b, id_from_node_id(self.id, scx), "{}", self.ident.to_string());
                text!(b, ": ");
                ty.make(b, id, scx)?;
                text!(b, ";");
            }
            ast::ForeignItemKind::Ty => {
                text!(b, "type ");
                // FIXME generics?
                def_!(b, id_from_node_id(self.id, scx), "{}", self.ident.to_string());
            }
            ast::ForeignItemKind::Macro(..) => Err("macro")?,
        }
        Ok(())
    }
}

fn name_and_generics(
    b: &mut SigBuilder,
    generics: &ast::Generics,
    id: NodeId,
    name: ast::Ident,
    scx: &SaveContext<'_, '_>,
) -> Result {
    let name = name.to_string();
    let id_ = id_from_node_id(id, scx);
    def_!(b, id_, "{}", name);
    generics.make(b, Some(id), scx)?;
    Ok(())
}

fn make_assoc_type_signature(
    b: &mut SigBuilder,
    id: NodeId,
    ident: ast::Ident,
    bounds: Option<&ast::GenericBounds>,
    default: Option<&ast::Ty>,
    scx: &SaveContext<'_, '_>,
) -> Result {
    text!(b, "type ");
    def_!(b, id_from_node_id(id, scx), "{}", ident.to_string());

    if let Some(bounds) = bounds {
        // FIXME should descend into bounds
        text!(b, ": {}", pprust::bounds_to_string(bounds));
    }
    if let Some(default) = default {
        text!(b, " = ");
        default.make(b, Some(id), scx)?;
    }
    text!(b, ";");
    Ok(())
}

fn make_assoc_const_signature(
    b: &mut SigBuilder,
    id: NodeId,
    ident: ast::Name,
    ty: &ast::Ty,
    default: Option<&ast::Expr>,
    scx: &SaveContext<'_, '_>,
) -> Result {
    text!(b, "const ");
    def_!(b, id_from_node_id(id, scx), "{}", ident.to_string());
    text!(b, ": ");
    ty.make(b, Some(id), scx)?;

    if let Some(default) = default {
        text!(b, " = {}", pprust::expr_to_string(default));
    }
    text!(b, ";");
    Ok(())
}

fn make_method_signature(
    b: &mut SigBuilder,
    id: NodeId,
    ident: ast::Ident,
    generics: &ast::Generics,
    m: &ast::MethodSig,
    scx: &SaveContext<'_, '_>,
) -> Result {
    m.header.make(b, Some(id), scx)?;

    name_and_generics(b, generics, id, ident, scx)?;

    text!(b, "(");

    let mut inputs = &m.decl.inputs[..];

    if inputs.len() > 0 {
        match inputs[0].ty.node {
            ast::TyKind::ImplicitSelf => {
                // [ref] [mut] self
                // FIXME descend into pattern
                text!(b, "{}, ", pprust::pat_to_string(&inputs[0].pat));
                inputs = &inputs[1..];
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                if let ast::TyKind::ImplicitSelf = mt.ty.node {
                    text!(b, "&");
                    if let &Some(ref l) = lifetime {
                        // FIXME id?
                        text!(b, "{} ", l.ident);
                    }
                    if let ast::Mutability::Mutable = mt.mutbl {
                        text!(b, "mut ");
                    };

                    text!(b, "self, ");
                    inputs = &inputs[1..];
                }
            } // FIXME need to handle self: Pin<&Self> here?? dont think so
            _ => {}
        }
    }

    for i in inputs {
        // FIXME should descend into patterns to add defs.
        text!(b, "{}:", pprust::pat_to_string(&i.pat));
        i.ty.make(b, Some(i.id), scx)?;
        // FIXME trailing comma
        text!(b, ", ");
    }

    text!(b, ")");

    if let ast::FunctionRetTy::Ty(ref t) = m.decl.output {
        text!(b, " -> ");
        t.make(b, None, scx)?;
    }
    text!(b, " {{}}");

    Ok(())
}
