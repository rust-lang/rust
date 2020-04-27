//! The compiler code necessary to implement the `#[derive(RustcEncodable)]`
//! (and `RustcDecodable`, in `decodable.rs`) extension. The idea here is that
//! type-defining items may be tagged with
//! `#[derive(RustcEncodable, RustcDecodable)]`.
//!
//! For example, a type like:
//!
//! ```
//! #[derive(RustcEncodable, RustcDecodable)]
//! struct Node { id: usize }
//! ```
//!
//! would generate two implementations like:
//!
//! ```
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
//! ```
//! # #[derive(RustcEncodable, RustcDecodable)]
//! # struct Span;
//! #[derive(RustcEncodable, RustcDecodable)]
//! struct Spanned<T> { node: T, span: Span }
//! ```
//!
//! would yield functions like:
//!
//! ```
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

use rustc_ast::ptr::P;
use rustc_ast::{Expr, ExprKind, MetaItem, Mutability};
use rustc_expand::base::{Annotatable, ExtCtxt};
use rustc_span::symbol::{sym, Ident, Symbol};
use rustc_span::Span;

pub fn expand_deriving_rustc_encodable(
    cx: &mut ExtCtxt<'_>,
    span: Span,
    mitem: &MetaItem,
    item: &Annotatable,
    push: &mut dyn FnMut(Annotatable),
) {
    let krate = sym::rustc_serialize;
    let typaram = sym::__S;

    let trait_def = TraitDef {
        span,
        attributes: Vec::new(),
        path: Path::new_(vec![krate, sym::Encodable], None, vec![], PathKind::Global),
        additional_bounds: Vec::new(),
        generics: Bounds::empty(),
        is_unsafe: false,
        supports_unions: false,
        methods: vec![MethodDef {
            name: sym::encode,
            generics: Bounds {
                bounds: vec![(
                    typaram,
                    vec![Path::new_(vec![krate, sym::Encoder], None, vec![], PathKind::Global)],
                )],
            },
            explicit_self: borrowed_explicit_self(),
            args: vec![(
                Ptr(Box::new(Literal(Path::new_local(typaram))), Borrowed(None, Mutability::Mut)),
                // FIXME: we could use `sym::s` here, but making `s` a static
                // symbol changes the symbol index ordering in a way that makes
                // ui/lint/rfc-2457-non-ascii-idents/lint-confusable-idents.rs
                // fail. The linting code should be fixed so that its output
                // does not depend on the symbol index ordering.
                Symbol::intern("s"),
            )],
            ret_ty: Literal(Path::new_(
                pathvec_std!(result::Result),
                None,
                vec![
                    Box::new(Tuple(Vec::new())),
                    Box::new(Literal(Path::new_(
                        vec![typaram, sym::Error],
                        None,
                        vec![],
                        PathKind::Local,
                    ))),
                ],
                PathKind::Std,
            )),
            attributes: Vec::new(),
            is_unsafe: false,
            unify_fieldless_variants: false,
            combine_substructure: combine_substructure(Box::new(|a, b, c| {
                encodable_substructure(a, b, c, krate)
            })),
        }],
        associated_types: Vec::new(),
    };

    trait_def.expand(cx, mitem, item, push)
}

fn encodable_substructure(
    cx: &mut ExtCtxt<'_>,
    trait_span: Span,
    substr: &Substructure<'_>,
    krate: Symbol,
) -> P<Expr> {
    let encoder = substr.nonself_args[0].clone();
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

    match *substr.fields {
        Struct(_, ref fields) => {
            let emit_struct_field = Ident::new(sym::emit_struct_field, trait_span);
            let mut stmts = Vec::new();
            for (i, &FieldInfo { name, ref self_, span, .. }) in fields.iter().enumerate() {
                let name = match name {
                    Some(id) => id.name,
                    None => Symbol::intern(&format!("_field{}", i)),
                };
                let self_ref = cx.expr_addr_of(span, self_.clone());
                let enc = cx.expr_call(span, fn_path.clone(), vec![self_ref, blkencoder.clone()]);
                let lambda = cx.lambda1(span, enc, blkarg);
                let call = cx.expr_method_call(
                    span,
                    blkencoder.clone(),
                    emit_struct_field,
                    vec![cx.expr_str(span, name), cx.expr_usize(span, i), lambda],
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
                let ok = cx.expr_ok(trait_span, cx.expr_tuple(trait_span, vec![]));
                cx.lambda1(trait_span, ok, blkarg)
            } else {
                cx.lambda_stmts_1(trait_span, stmts, blkarg)
            };

            cx.expr_method_call(
                trait_span,
                encoder,
                Ident::new(sym::emit_struct, trait_span),
                vec![
                    cx.expr_str(trait_span, substr.type_ident.name),
                    cx.expr_usize(trait_span, fields.len()),
                    blk,
                ],
            )
        }

        EnumMatching(idx, _, variant, ref fields) => {
            // We're not generating an AST that the borrow checker is expecting,
            // so we need to generate a unique local variable to take the
            // mutable loan out on, otherwise we get conflicts which don't
            // actually exist.
            let me = cx.stmt_let(trait_span, false, blkarg, encoder);
            let encoder = cx.expr_ident(trait_span, blkarg);
            let emit_variant_arg = Ident::new(sym::emit_enum_variant_arg, trait_span);
            let mut stmts = Vec::new();
            if !fields.is_empty() {
                let last = fields.len() - 1;
                for (i, &FieldInfo { ref self_, span, .. }) in fields.iter().enumerate() {
                    let self_ref = cx.expr_addr_of(span, self_.clone());
                    let enc =
                        cx.expr_call(span, fn_path.clone(), vec![self_ref, blkencoder.clone()]);
                    let lambda = cx.lambda1(span, enc, blkarg);
                    let call = cx.expr_method_call(
                        span,
                        blkencoder.clone(),
                        emit_variant_arg,
                        vec![cx.expr_usize(span, i), lambda],
                    );
                    let call = if i != last {
                        cx.expr_try(span, call)
                    } else {
                        cx.expr(span, ExprKind::Ret(Some(call)))
                    };
                    stmts.push(cx.stmt_expr(call));
                }
            } else {
                let ok = cx.expr_ok(trait_span, cx.expr_tuple(trait_span, vec![]));
                let ret_ok = cx.expr(trait_span, ExprKind::Ret(Some(ok)));
                stmts.push(cx.stmt_expr(ret_ok));
            }

            let blk = cx.lambda_stmts_1(trait_span, stmts, blkarg);
            let name = cx.expr_str(trait_span, variant.ident.name);
            let call = cx.expr_method_call(
                trait_span,
                blkencoder,
                Ident::new(sym::emit_enum_variant, trait_span),
                vec![
                    name,
                    cx.expr_usize(trait_span, idx),
                    cx.expr_usize(trait_span, fields.len()),
                    blk,
                ],
            );
            let blk = cx.lambda1(trait_span, call, blkarg);
            let ret = cx.expr_method_call(
                trait_span,
                encoder,
                Ident::new(sym::emit_enum, trait_span),
                vec![cx.expr_str(trait_span, substr.type_ident.name), blk],
            );
            cx.expr_block(cx.block(trait_span, vec![me, cx.stmt_expr(ret)]))
        }

        _ => cx.bug("expected Struct or EnumMatching in derive(Encodable)"),
    }
}
