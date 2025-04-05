use std::fmt;

use askama::Template;
use rustc_abi::{Primitive, TagEncoding, Variants};
use rustc_hir::def_id::DefId;
use rustc_middle::span_bug;
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{self};
use rustc_span::symbol::Symbol;

use crate::html::render::Context;

#[derive(Template)]
#[template(path = "type_layout.html")]
struct TypeLayout<'cx> {
    variants: Vec<(Symbol, TypeLayoutSize)>,
    type_layout_size: Result<TypeLayoutSize, &'cx LayoutError<'cx>>,
}

#[derive(Template)]
#[template(path = "type_layout_size.html")]
struct TypeLayoutSize {
    is_unsized: bool,
    is_uninhabited: bool,
    size: u64,
}

pub(crate) fn document_type_layout(cx: &Context<'_>, ty_def_id: DefId) -> impl fmt::Display {
    fmt::from_fn(move |f| {
        if !cx.shared.show_type_layout {
            return Ok(());
        }

        let tcx = cx.tcx();
        let typing_env = ty::TypingEnv::post_analysis(tcx, ty_def_id);
        let ty = tcx.type_of(ty_def_id).instantiate_identity();
        let type_layout = tcx.layout_of(typing_env.as_query_input(ty));

        let variants = if let Ok(type_layout) = type_layout
            && let Variants::Multiple { variants, tag, tag_encoding, .. } =
                type_layout.layout.variants()
            && !variants.is_empty()
        {
            let tag_size = if let TagEncoding::Niche { .. } = tag_encoding {
                0
            } else if let Primitive::Int(i, _) = tag.primitive() {
                i.size().bytes()
            } else {
                span_bug!(tcx.def_span(ty_def_id), "tag is neither niche nor int")
            };
            variants
                .iter_enumerated()
                .map(|(variant_idx, variant_layout)| {
                    let ty::Adt(adt, _) = type_layout.ty.kind() else {
                        span_bug!(tcx.def_span(ty_def_id), "not an adt")
                    };
                    let name = adt.variant(variant_idx).name;
                    let is_unsized = variant_layout.is_unsized();
                    let is_uninhabited = variant_layout.is_uninhabited();
                    let size = variant_layout.size.bytes() - tag_size;
                    let type_layout_size = TypeLayoutSize { is_unsized, is_uninhabited, size };
                    (name, type_layout_size)
                })
                .collect()
        } else {
            Vec::new()
        };

        let type_layout_size = tcx.layout_of(typing_env.as_query_input(ty)).map(|layout| {
            let is_unsized = layout.is_unsized();
            let is_uninhabited = layout.is_uninhabited();
            let size = layout.size.bytes();
            TypeLayoutSize { is_unsized, is_uninhabited, size }
        });

        TypeLayout { variants, type_layout_size }.render_into(f).unwrap();
        Ok(())
    })
}
