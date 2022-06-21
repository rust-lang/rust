use crate::clean::{inline::print_inlined_const, utils::print_const_expr};
use crate::formats::{cache::Cache, item_type::ItemType};
use crate::html::{escape::Escape, format::Buffer};
use rustc_hir::def::{CtorKind, DefKind};
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::{AllocRange, ConstValue, Scalar};
use rustc_middle::mir::ConstantKind;
use rustc_middle::ty::{self, util::is_doc_hidden};
use rustc_middle::ty::{Const, ConstInt, DefIdTree, FieldDef, ParamConst, ScalarInt};
use rustc_middle::ty::{Ty, TyCtxt, TypeVisitable, Visibility};
use rustc_span::sym;
use rustc_target::abi::Size;
use std::fmt::Write;

/// Try to evaluate the given constant expression and render it as HTML code or as plain text.
///
/// `None` is returned if the expression is “too generic”.
///
/// This textual representation may not actually be lossless or even valid
/// Rust syntax since
///
/// * overly long arrays & strings and deeply-nested subexpressions are replaced
///   with ellipses
/// * private and `#[doc(hidden)]` struct fields are omitted
/// * `..` is added to (tuple) struct literals if any fields are omitted
///   or if the type is `#[non_exhaustive]`
/// * `_` is used in place of “unsupported” expressions (e.g. pointers)
pub(crate) fn eval_and_render_const(def_id: DefId, renderer: &Renderer<'_, '_>) -> Option<String> {
    let value = renderer.tcx().const_eval_poly(def_id).ok()?;
    let mut buffer = renderer.buffer();
    render_const_value(&mut buffer, value, renderer.tcx().type_of(def_id), renderer, 0);
    Some(buffer.into_inner())
}

pub(crate) enum Renderer<'cx, 'tcx> {
    PlainText(crate::json::Context<'tcx>),
    Html(&'cx super::Context<'tcx>),
}

impl<'cx, 'tcx> Renderer<'cx, 'tcx> {
    fn buffer(&self) -> Buffer {
        match self {
            Self::PlainText(_) => Buffer::new(),
            Self::Html(_) => Buffer::html(),
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        match self {
            Self::PlainText(cx) => cx.tcx,
            Self::Html(cx) => cx.tcx(),
        }
    }

    fn cache(&self) -> &Cache {
        match self {
            Self::PlainText(cx) => &cx.cache,
            Self::Html(cx) => &cx.shared.cache,
        }
    }
}

const DEPTH_LIMIT: u32 = 2;
const STRING_LENGTH_LIMIT: usize = 80;
const ARRAY_LENGTH_LIMIT: usize = 12;

const DEPTH_ELLIPSIS: &str = "…";
const LENGTH_ELLIPSIS: &str = "………";

fn render_constant_kind<'tcx>(
    buffer: &mut Buffer,
    ct: ConstantKind<'tcx>,
    renderer: &Renderer<'_, 'tcx>,
    depth: u32,
) {
    if depth > DEPTH_LIMIT {
        render_ellipsis(buffer, DEPTH_ELLIPSIS);
        return;
    }

    match ct {
        ConstantKind::Ty(ct) => render_const(buffer, ct, renderer),
        ConstantKind::Val(ct, ty) => render_const_value(buffer, ct, ty, renderer, depth),
    }
}

fn render_const<'tcx>(buffer: &mut Buffer, ct: Const<'tcx>, renderer: &Renderer<'_, 'tcx>) {
    let tcx = renderer.tcx();

    match ct.kind() {
        ty::ConstKind::Unevaluated(ty::Unevaluated { def, promoted: Some(promoted), .. }) => {
            render_path(buffer, def.did, renderer);
            write!(buffer, "::{:?}", promoted);
        }
        ty::ConstKind::Unevaluated(ty::Unevaluated { def, promoted: None, .. }) => {
            match tcx.def_kind(def.did) {
                DefKind::Static(..) | DefKind::Const | DefKind::AssocConst => {
                    render_path(buffer, def.did, renderer)
                }
                _ => {
                    let expr = match def.as_local() {
                        Some(def) => print_const_expr(tcx, tcx.hir().body_owned_by(def.did)),
                        None => print_inlined_const(tcx, def.did),
                    };

                    if buffer.is_for_html() {
                        write!(buffer, "{}", Escape(&expr));
                    } else {
                        write!(buffer, "{expr}");
                    }
                }
            }
        }
        ty::ConstKind::Param(ParamConst { name, .. }) => write!(buffer, "{name}"),
        ty::ConstKind::Value(value) => render_valtree(buffer, value, ct.ty()),
        ty::ConstKind::Infer(_)
        | ty::ConstKind::Bound(..)
        | ty::ConstKind::Placeholder(_)
        | ty::ConstKind::Error(_) => write!(buffer, "_"),
    }
}

fn render_const_value<'tcx>(
    buffer: &mut Buffer,
    ct: ConstValue<'tcx>,
    ty: Ty<'tcx>,
    renderer: &Renderer<'_, 'tcx>,
    depth: u32,
) {
    let tcx = renderer.tcx();
    // FIXME: The code inside `rustc_middle::mir::pretty_print_const` does this.
    //        Do we need to do this, too? Why (not)?
    // let ct = tcx.lift(ct).unwrap();
    // let ty = tcx.lift(ty).unwrap();
    let u8_type = tcx.types.u8;

    match (ct, ty.kind()) {
        (ConstValue::Slice { data, start, end }, ty::Ref(_, inner, _)) => {
            match inner.kind() {
                ty::Slice(t) if *t == u8_type => {
                    // The `inspect` here is okay since we checked the bounds, and there are
                    // no relocations (we have an active slice reference here). We don't use
                    // this result to affect interpreter execution.
                    let byte_str =
                        data.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..end);
                    render_byte_str(buffer, byte_str);
                }
                ty::Str => {
                    // The `inspect` here is okay since we checked the bounds, and there are no
                    // relocations (we have an active `str` reference here). We don't use this
                    // result to affect interpreter execution.
                    let slice =
                        data.inner().inspect_with_uninit_and_ptr_outside_interpreter(start..end);

                    // FIXME: Make the limit depend on the `depth` (inversely proportionally)
                    if slice.len() > STRING_LENGTH_LIMIT {
                        write!(buffer, "\"");
                        render_ellipsis(buffer, LENGTH_ELLIPSIS);
                        write!(buffer, "\"");
                    } else {
                        let slice = format!("{:?}", String::from_utf8_lossy(slice));

                        if buffer.is_for_html() {
                            write!(buffer, "{}", Escape(&slice));
                        } else {
                            write!(buffer, "{slice}");
                        }
                    }
                }
                // `ConstValue::Slice` is only used for `&[u8]` and `&str`.
                _ => unreachable!(),
            }
        }
        (ConstValue::ByRef { alloc, offset }, ty::Array(t, n)) if *t == u8_type => {
            let n = n.kind().try_to_bits(tcx.data_layout.pointer_size).unwrap();
            // cast is ok because we already checked for pointer size (32 or 64 bit) above
            let range = AllocRange { start: offset, size: Size::from_bytes(n) };
            let byte_str = alloc.inner().get_bytes(&tcx, range).unwrap();
            write!(buffer, "*");
            render_byte_str(buffer, byte_str);
        }
        // Aggregates.
        //
        // NB: the `has_param_types_or_consts` check ensures that we can use
        // the `destructure_const` query with an empty `ty::ParamEnv` without
        // introducing ICEs (e.g. via `layout_of`) from missing bounds.
        // E.g. `transmute([0usize; 2]): (u8, *mut T)` needs to know `T: Sized`
        // to be able to destructure the tuple into `(0u8, *mut T)
        (_, ty::Array(..) | ty::Tuple(..) | ty::Adt(..)) if !ty.has_param_types_or_consts() => {
            // FIXME: The code inside `rustc_middle::mir::pretty_print_const` does this.
            //        Do we need to do this, too? Why (not)?
            // let ct = tcx.lift(ct).unwrap();
            // let ty = tcx.lift(ty).unwrap();
            let Some(contents) = tcx.try_destructure_mir_constant(
                ty::ParamEnv::reveal_all().and(ConstantKind::Val(ct, ty))
            ) else {
                return write!(buffer, "_");
            };

            let mut fields = contents.fields.iter().copied();

            // FIXME: Should we try to print larger structs etc. across multiple lines?
            match *ty.kind() {
                ty::Array(..) => {
                    write!(buffer, "[");

                    // FIXME: Make the limit depend on the `depth` (inversely proportionally)
                    if contents.fields.len() > ARRAY_LENGTH_LIMIT {
                        render_ellipsis(buffer, LENGTH_ELLIPSIS);
                    } else if let Some(first) = fields.next() {
                        render_constant_kind(buffer, first, renderer, depth + 1);
                        for field in fields {
                            buffer.write_str(", ");
                            render_constant_kind(buffer, field, renderer, depth + 1);
                        }
                    }

                    write!(buffer, "]");
                }
                ty::Tuple(..) => {
                    write!(buffer, "(");

                    if let Some(first) = fields.next() {
                        render_constant_kind(buffer, first, renderer, depth + 1);
                        for field in fields {
                            buffer.write_str(", ");
                            render_constant_kind(buffer, field, renderer, depth + 1);
                        }
                    }
                    if contents.fields.len() == 1 {
                        write!(buffer, ",");
                    }

                    write!(buffer, ")");
                }
                ty::Adt(def, _) if !def.variants().is_empty() => {
                    let should_hide = |field: &FieldDef| {
                        // FIXME: Should I use `cache.access_levels.is_public(did)` here instead?
                        is_doc_hidden(tcx, field.did)
                            && !(renderer.cache().document_hidden && field.did.is_local())
                            || field.vis != Visibility::Public
                                && !(renderer.cache().document_private && field.did.is_local())
                    };

                    let is_non_exhaustive = tcx.has_attr(def.did(), sym::non_exhaustive);

                    let variant_idx =
                        contents.variant.expect("destructed const of adt without variant idx");
                    let variant_def = &def.variant(variant_idx);
                    render_path(buffer, variant_def.def_id, renderer);

                    match variant_def.ctor_kind {
                        CtorKind::Const => {
                            if is_non_exhaustive {
                                write!(buffer, " {{ .. }}");
                            }
                        }
                        CtorKind::Fn => {
                            write!(buffer, "(");

                            let mut first = true;
                            for (field_def, field) in std::iter::zip(&variant_def.fields, fields) {
                                if !first {
                                    write!(buffer, ", ");
                                }
                                first = false;

                                if should_hide(field_def) {
                                    write!(buffer, "_");
                                    continue;
                                }

                                render_constant_kind(buffer, field, renderer, depth + 1);
                            }

                            if is_non_exhaustive {
                                if !first {
                                    write!(buffer, ", ");
                                }

                                // Using `..` (borrowed from patterns) to mark non-exhaustive tuple
                                // structs in our pseudo-Rust expression syntax is sadly not without
                                // caveats since in real-Rust expressions, it denotes full ranges
                                // (`std::ops::RangeFull`) which may appear as arguments to tuple
                                // struct constructors (albeit incredibly uncommonly) and thus
                                // it may lead to confusion.
                                // NB: Actually we literally render full ranges as `RangeFull`.
                                //     Still, that does not help that much.
                                // If this issue turns out to be significant, we can change the
                                // output to e.g. `_ @ ..` (borrowed from slice patterns) which is
                                // not a valid Rust expression at the time of this writing but
                                // it's quite cryptic.
                                write!(buffer, "..");
                            }

                            write!(buffer, ")");
                        }
                        CtorKind::Fictive => {
                            write!(buffer, " {{ ");

                            let mut first = true;
                            let mut did_hide_fields = false;
                            for (field_def, field) in std::iter::zip(&variant_def.fields, fields) {
                                if should_hide(field_def) {
                                    did_hide_fields = true;
                                    continue;
                                }

                                if !first {
                                    write!(buffer, ", ");
                                }
                                first = false;

                                render_field_name(buffer, field_def, renderer);
                                write!(buffer, ": ");
                                render_constant_kind(buffer, field, renderer, depth + 1);
                            }

                            if did_hide_fields || is_non_exhaustive {
                                if !first {
                                    write!(buffer, ", ");
                                }
                                write!(buffer, "..");
                            }

                            write!(buffer, " }}");
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        (ConstValue::Scalar(Scalar::Int(int)), _) => render_const_scalar_int(buffer, int, ty),
        // FIXME: Support `&[_]`: `(ByRef { .. }, ty::Ref(_, ty, Not)) if let ty::Slice(_) = ty`
        //        Blocker: try_destructure_mir_constant does not support slices.
        _ => write!(buffer, "_"),
    }
}

fn render_valtree<'tcx>(buffer: &mut Buffer, _valtree: ty::ValTree<'tcx>, _ty: Ty<'tcx>) {
    // FIXME: If this case is actually reachable, adopt the code from
    //        rustc_middle::ty::print::pretty::PrettyPrinter::pretty_print_const_valtree
    write!(buffer, "_");
}

fn render_path<'tcx>(buffer: &mut Buffer, def_id: DefId, renderer: &Renderer<'_, 'tcx>) {
    let tcx = renderer.tcx();
    let name = tcx.item_name(def_id);

    match renderer {
        Renderer::PlainText(_) => write!(buffer, "{name}"),
        Renderer::Html(cx) => {
            if let Ok((mut url, item_type, path)) = super::href(def_id, cx) {
                let mut needs_fragment = true;
                let item_type = match tcx.def_kind(def_id) {
                    DefKind::AssocFn => {
                        if tcx.associated_item(def_id).defaultness(tcx).has_value() {
                            ItemType::Method
                        } else {
                            ItemType::TyMethod
                        }
                    }
                    DefKind::AssocTy => ItemType::AssocType,
                    DefKind::AssocConst => ItemType::AssocConst,
                    DefKind::Variant => ItemType::Variant,
                    _ => {
                        needs_fragment = false;
                        item_type
                    }
                };

                let mut path = super::join_with_double_colon(&path);

                if needs_fragment {
                    write!(url, "#{item_type}.{name}").unwrap();
                    write!(path, "::{name}").unwrap();
                }

                write!(
                    buffer,
                    r#"<a class="{item_type}" href="{url}" title="{item_type} {path}">{name}</a>"#,
                );
            } else {
                write!(buffer, "{name}");
            }
        }
    }
}

fn render_byte_str(buffer: &mut Buffer, byte_str: &[u8]) {
    buffer.write_str("b\"");

    // FIXME: Make the limit depend on the `depth` (inversely proportionally)
    if byte_str.len() > STRING_LENGTH_LIMIT {
        render_ellipsis(buffer, LENGTH_ELLIPSIS);
    } else {
        for &char in byte_str {
            for char in std::ascii::escape_default(char) {
                let char = char::from(char).to_string();

                if buffer.is_for_html() {
                    write!(buffer, "{}", Escape(&char));
                } else {
                    write!(buffer, "{char}");
                }
            }
        }
    }

    buffer.write_str("\"");
}

fn render_const_scalar_int<'tcx>(buffer: &mut Buffer, int: ScalarInt, ty: Ty<'tcx>) {
    extern crate rustc_apfloat;
    use rustc_apfloat::ieee::{Double, Single};

    match ty.kind() {
        ty::Bool if int == ScalarInt::FALSE => write!(buffer, "false"),
        ty::Bool if int == ScalarInt::TRUE => write!(buffer, "true"),

        ty::Float(ty::FloatTy::F32) => {
            write!(buffer, "{}", Single::try_from(int).unwrap());
        }
        ty::Float(ty::FloatTy::F64) => {
            write!(buffer, "{}", Double::try_from(int).unwrap());
        }

        ty::Uint(_) | ty::Int(_) => {
            let int =
                ConstInt::new(int, matches!(ty.kind(), ty::Int(_)), ty.is_ptr_sized_integral());
            // FIXME: We probably shouldn't use the *Debug* impl for *user-facing output*.
            //        However, it looks really nice and its implementation is non-trivial.
            //        Should we modify rustc_middle and make it a *Display* impl?
            write!(buffer, "{int:?}");
        }
        ty::Char if char::try_from(int).is_ok() => {
            // FIXME: We probably shouldn't use the *Debug* impl here (see fixme above).
            write!(buffer, "{:?}", char::try_from(int).unwrap());
        }
        _ => write!(buffer, "_"),
    }
}

fn render_field_name(buffer: &mut Buffer, field_def: &FieldDef, renderer: &Renderer<'_, '_>) {
    let tcx = renderer.tcx();

    match renderer {
        Renderer::PlainText(_) => write!(buffer, "{}", field_def.name),
        Renderer::Html(cx) => match super::href(field_def.did, cx) {
            Ok((mut url, ..)) => {
                write!(url, "#").unwrap();
                let parent_id = tcx.parent(field_def.did);
                if tcx.def_kind(parent_id) == DefKind::Variant {
                    write!(url, "{}.{}.field", ItemType::Variant, tcx.item_name(parent_id))
                } else {
                    write!(url, "{}", ItemType::StructField)
                }
                .unwrap();

                write!(url, ".{}", field_def.name).unwrap();

                write!(
                    buffer,
                    r#"<a class="{}" href="{}" title="field {}">{}</a>"#,
                    ItemType::StructField,
                    url,
                    field_def.name,
                    field_def.name,
                );
            }
            Err(_) => write!(buffer, "{}", field_def.name),
        },
    }
}

fn render_ellipsis(buffer: &mut Buffer, ellipsis: &str) {
    if buffer.is_for_html() {
        write!(buffer, r#"<span class="ellipsis">"#);
    }

    write!(buffer, "{ellipsis}");

    if buffer.is_for_html() {
        write!(buffer, "</span>");
    }
}
