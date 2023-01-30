//! The compiler code necessary to implement the `#[derive(RustcEncodable)]`
//! (and `RustcDecodable`, in `decodable.rs`) extension. The idea here is that
//! type-defining items may be tagged with
//! `#[derive(RustcEncodable, RustcDecodable)]`.
//!
//! For example, a type like:
//!
//! ```ignore (old code)
//! #[derive(RustcEncodable, RustcDecodable)]
//! struct Node { id: usize }
//! ```
//!
//! would generate two implementations like:
//!
//! ```ignore (old code)
//! # struct Node { id: usize }
//! impl<S: Encoder<E>, E> Encodable<S, E> for Node {
//!     fn encode(&self, s: &mut S) -> Result<(), E> {
//!         s.emit_struct("Node", 1, |this| {
//!             this.emit_struct_field("id", 0, |this| {
//!                 Encodable::encode(&self.id, this)
//!                 /* this.emit_usize(self.id) can also be used */
//!             })
//!         })
//!     }
//! }
//!
//! impl<D: Decoder<E>, E> Decodable<D, E> for Node {
//!     fn decode(d: &mut D) -> Result<Node, E> {
//!         d.read_struct("Node", 1, |this| {
//!             match this.read_struct_field("id", 0, |this| Decodable::decode(this)) {
//!                 Ok(id) => Ok(Node { id: id }),
//!                 Err(e) => Err(e),
//!             }
//!         })
//!     }
//! }
//! ```
//!
//! Other interesting scenarios are when the item has type parameters or
//! references other non-built-in types. A type definition like:
//!
//! ```ignore (old code)
//! # #[derive(RustcEncodable, RustcDecodable)]
//! # struct Span;
//! #[derive(RustcEncodable, RustcDecodable)]
//! struct Spanned<T> { node: T, span: Span }
//! ```
//!
//! would yield functions like:
//!
//! ```ignore (old code)
//! # #[derive(RustcEncodable, RustcDecodable)]
//! # struct Span;
//! # struct Spanned<T> { node: T, span: Span }
//! impl<
//!     S: Encoder<E>,
//!     E,
//!     T: Encodable<S, E>
//! > Encodable<S, E> for Spanned<T> {
//!     fn encode(&self, s: &mut S) -> Result<(), E> {
//!         s.emit_struct("Spanned", 2, |this| {
//!             this.emit_struct_field("node", 0, |this| self.node.encode(this))
//!                 .unwrap();
//!             this.emit_struct_field("span", 1, |this| self.span.encode(this))
//!         })
//!     }
//! }
//!
//! impl<
//!     D: Decoder<E>,
//!     E,
//!     T: Decodable<D, E>
//! > Decodable<D, E> for Spanned<T> {
//!     fn decode(d: &mut D) -> Result<Spanned<T>, E> {
//!         d.read_struct("Spanned", 2, |this| {
//!             Ok(Spanned {
//!                 node: this.read_struct_field("node", 0, |this| Decodable::decode(this))
//!                     .unwrap(),
//!                 span: this.read_struct_field("span", 1, |this| Decodable::decode(this))
//!                     .unwrap(),
//!             })
//!         })
//!     }
//! }
//! ```

use crate::deriving::generic::ty::*;
use crate::deriving::generic::*;
use crate::deriving::pathvec_std;
use rustc_ast::{AttrVec, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;
use thin_vec::{thin_vec, ThinVec};

pub fn expand_deriving_rustc_encodable(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
    is_const: bool,
) {
    let krate = sym::rustc_serialize;
    let typaram = sym::__S;

    let trait_def = TraitDef {
        span,
        path: Path::new_(vec![krate, sym::Encodable], vec![], PathKind::Global),
        skip_path_as_bound: false,
        needs_copy_as_bound_if_packed: true,
        additional_bounds: Vec::new(),
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::encode,
            generics: Bounds {
                bounds: vec![(
                    typaram,
                    vec![Path::new_(vec![krate, sym::Encoder], vec![], PathKind::Global)],
                )],
            },
            explicit_self: true,
            nonself_args: vec![(
                Ref(Box::new(Path(Path::new_local(typaram))), Mutability::Mut),
                sym::s,
            )],
            ret_ty: Path(Path::new_(
                pathvec_std!(result::Result),
                vec![
                    Box::new(Unit),
                    Box::new(Path(Path::new_(vec![typaram, sym::Error], vec![], PathKind::Local))),
                ],
                PathKind::Std,
            )),
            attributes: AttrVec::new(),
            fieldless_variants_strategy: FieldlessVariantsStrategy::Default,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                encodable_substructure(a, b, c, krate)
            })),
        }],
        associated_types: Vec::new(),
        is_const,
    };

    trait_def.expand(cx, mitem, item, push)
}

fn encodable_substructure(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
    krate: Symbol,
) -> BlockOrExpr {
    let encoder = substr.nonselflike_args[0].clone();
    // throw an underscore in front to suppress unused variable warnings
    let blkarg = Ident::new(sym::_e, trait_span);
    let blkencoder = cx.expr_ident(trait_span, blkarg);
    let fn_path = cx.expr_path(cx.path_global(
        trait_span,
        vec![
            Ident::new(krate, trait_span),
            Ident::new(sym::Encodable, trait_span),
            Ident::new(sym::encode, trait_span),
        ],
    ));

    match substr.fields {
        Struct(_, fields) => {
            let fn_emit_struct_field_path =
                cx.def_site_path(&[sym::rustc_serialize, sym::Encoder, sym::emit_struct_field]);
            let mut stmts = ThinVec::new();
            for (i, &FieldInfo { name, ref self_expr, span, .. }) in fields.iter().enumerate() {
                let name = match name {
                    Some(id) => id.name,
                    None => Symbol::intern(&format!("_field{}", i)),
                };
                let self_ref = cx.expr_addr_of(span, self_expr.clone());
                let enc =
                    cx.expr_call(span, fn_path.clone(), thin_vec![self_ref, blkencoder.clone()]);
                let lambda = cx.lambda1(span, enc, blkarg);
                let call = cx.expr_call_global(
                    span,
                    fn_emit_struct_field_path.clone(),
                    thin_vec![
                        blkencoder.clone(),
                        cx.expr_str(span, name),
                        cx.expr_usize(span, i),
                        lambda,
                    ],
                );

                // last call doesn't need a try!
                let last = fields.len() - 1;
                let call = if i != last {
                    cx.expr_try(span, call)
                } else {
                    cx.expr(span, ExprKind::Ret(Some(call)))
                };

                let stmt = cx.stmt_expr(call);
                stmts.push(stmt);
            }

            // unit structs have no fields and need to return Ok()
            let blk = if stmts.is_empty() {
                let ok = cx.expr_ok(trait_span, cx.expr_tuple(trait_span, ThinVec::new()));
                cx.lambda1(trait_span, ok, blkarg)
            } else {
                cx.lambda_stmts_1(trait_span, stmts, blkarg)
            };

            let fn_emit_struct_path =
                cx.def_site_path(&[sym::rustc_serialize, sym::Encoder, sym::emit_struct]);

            let expr = cx.expr_call_global(
                trait_span,
                fn_emit_struct_path,
                thin_vec![
                    encoder,
                    cx.expr_str(trait_span, substr.type_ident.name),
                    cx.expr_usize(trait_span, fields.len()),
                    blk,
                ],
            );
            BlockOrExpr::new_expr(expr)
        }

        EnumMatching(idx, _, variant, fields) => {
            // We're not generating an AST that the borrow checker is expecting,
            // so we need to generate a unique local variable to take the
            // mutable loan out on, otherwise we get conflicts which don't
            // actually exist.
            let me = cx.stmt_let(trait_span, false, blkarg, encoder);
            let encoder = cx.expr_ident(trait_span, blkarg);

            let fn_emit_enum_variant_arg_path: Vec<_> =
                cx.def_site_path(&[sym::rustc_serialize, sym::Encoder, sym::emit_enum_variant_arg]);

            let mut stmts = ThinVec::new();
            if !fields.is_empty() {
                let last = fields.len() - 1;
                for (i, &FieldInfo { ref self_expr, span, .. }) in fields.iter().enumerate() {
                    let self_ref = cx.expr_addr_of(span, self_expr.clone());
                    let enc = cx.expr_call(
                        span,
                        fn_path.clone(),
                        thin_vec![self_ref, blkencoder.clone()],
                    );
                    let lambda = cx.lambda1(span, enc, blkarg);

                    let call = cx.expr_call_global(
                        span,
                        fn_emit_enum_variant_arg_path.clone(),
                        thin_vec![blkencoder.clone(), cx.expr_usize(span, i), lambda],
                    );
                    let call = if i != last {
                        cx.expr_try(span, call)
                    } else {
                        cx.expr(span, ExprKind::Ret(Some(call)))
                    };
                    stmts.push(cx.stmt_expr(call));
                }
            } else {
                let ok = cx.expr_ok(trait_span, cx.expr_tuple(trait_span, ThinVec::new()));
                let ret_ok = cx.expr(trait_span, ExprKind::Ret(Some(ok)));
                stmts.push(cx.stmt_expr(ret_ok));
            }

            let blk = cx.lambda_stmts_1(trait_span, stmts, blkarg);
            let name = cx.expr_str(trait_span, variant.ident.name);

            let fn_emit_enum_variant_path: Vec<_> =
                cx.def_site_path(&[sym::rustc_serialize, sym::Encoder, sym::emit_enum_variant]);

            let call = cx.expr_call_global(
                trait_span,
                fn_emit_enum_variant_path,
                thin_vec![
                    blkencoder,
                    name,
                    cx.expr_usize(trait_span, *idx),
                    cx.expr_usize(trait_span, fields.len()),
                    blk,
                ],
            );

            let blk = cx.lambda1(trait_span, call, blkarg);
            let fn_emit_enum_path: Vec<_> =
                cx.def_site_path(&[sym::rustc_serialize, sym::Encoder, sym::emit_enum]);
            let expr = cx.expr_call_global(
                trait_span,
                fn_emit_enum_path,
                thin_vec![encoder, cx.expr_str(trait_span, substr.type_ident.name), blk],
            );
            BlockOrExpr::new_mixed(thin_vec![me], Some(expr))
        }

        _ => cx.bug("expected Struct or EnumMatching in derive(Encodable)"),
    }
}
