//@aux-build:paths.rs
#![deny(clippy::internal)]
#![feature(rustc_private)]

extern crate clippy_utils;
extern crate paths;
extern crate rustc_hir;
extern crate rustc_lint;
extern crate rustc_middle;
extern crate rustc_span;

#[allow(unused)]
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item, match_type};
#[allow(unused)]
use clippy_utils::{
    is_expr_path_def_path, is_path_diagnostic_item, is_res_diagnostic_ctor, is_res_lang_ctor, is_trait_method,
    match_def_path, match_trait_method, path_res,
};

#[allow(unused)]
use rustc_hir::LangItem;
#[allow(unused)]
use rustc_span::sym;

use rustc_hir::def_id::DefId;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;

#[allow(unused, clippy::unnecessary_def_path)]
static OPTION: [&str; 3] = ["core", "option", "Option"];
#[allow(unused, clippy::unnecessary_def_path)]
const RESULT: &[&str] = &["core", "result", "Result"];

fn _f<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, did: DefId, expr: &Expr<'_>) {
    let _ = match_type(cx, ty, &OPTION);
    let _ = match_type(cx, ty, RESULT);
    let _ = match_type(cx, ty, &["core", "result", "Result"]);

    #[allow(unused, clippy::unnecessary_def_path)]
    let rc_path = &["alloc", "rc", "Rc"];
    let _ = clippy_utils::ty::match_type(cx, ty, rc_path);

    let _ = match_type(cx, ty, &paths::OPTION);
    let _ = match_type(cx, ty, paths::RESULT);

    let _ = match_type(cx, ty, &["alloc", "boxed", "Box"]);
    let _ = match_type(cx, ty, &["core", "mem", "maybe_uninit", "MaybeUninit", "uninit"]);

    let _ = match_def_path(cx, did, &["alloc", "boxed", "Box"]);
    let _ = match_def_path(cx, did, &["core", "option", "Option"]);
    let _ = match_def_path(cx, did, &["core", "option", "Option", "Some"]);

    let _ = match_trait_method(cx, expr, &["core", "convert", "AsRef"]);

    let _ = is_expr_path_def_path(cx, expr, &["core", "option", "Option"]);
    let _ = is_expr_path_def_path(cx, expr, &["core", "iter", "traits", "Iterator", "next"]);
    let _ = is_expr_path_def_path(cx, expr, &["core", "option", "Option", "Some"]);
}

fn main() {}
