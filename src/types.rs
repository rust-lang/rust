

use syntax::ptr::P;
use syntax::ast;
use syntax::ast::*;
use rustc::lint::{Context, LintPass, LintArray, Lint, Level};
use syntax::codemap::Span;

/// Handles all the linting of funky types
pub struct TypePass;

declare_lint!(CLIPPY_BOX_VEC, Warn,
              "Warn on usage of Box<Vec<T>>")
declare_lint!(CLIPPY_DLIST, Warn,
              "Warn on usage of DList")

/// Matches a type with a provided string, and returns its type parameters if successful
pub fn match_ty_unwrap<'a>(ty: &'a Ty, segments: &[&str]) -> Option<&'a [P<Ty>]> {
    match ty.node {
        TyPath(Path {segments: ref seg, ..}, _, _) => {
            // So ast::Path isn't the full path, just the tokens that were provided.
            // I could muck around with the maps and find the full path
            // however the more efficient way is to simply reverse the iterators and zip them
            // which will compare them in reverse until one of them runs out of segments
            if seg.iter().rev().zip(segments.iter().rev()).all(|(a,b)| a.identifier.as_str() == *b) {
                match seg.as_slice().last() {
                    Some(&PathSegment {parameters: AngleBracketedParameters(ref a), ..}) => {
                        Some(a.types.as_slice())
                    }
                    _ => None
                }
            } else {
                None
            }
        },
        _ => None
    }
}

/// Lets me span a note only if the lint is shown
pub fn span_note_and_lint(cx: &Context, lint: &'static Lint, span: Span, msg: &str, note: &str) {
    cx.span_lint(lint, span, msg);
    if cx.current_level(lint) != Level::Allow {
        cx.sess().span_note(span, note);
    }
}

impl LintPass for TypePass {
    fn get_lints(&self) -> LintArray {
        lint_array!(CLIPPY_BOX_VEC, CLIPPY_DLIST)
    }

    fn check_ty(&mut self, cx: &Context, ty: &ast::Ty) {
        {
            // In case stuff gets moved around
            use std::boxed::Box;
            use std::vec::Vec;
        }
        match_ty_unwrap(ty, &["std", "boxed", "Box"]).and_then(|t| t.head())
          .map(|t| match_ty_unwrap(&**t, &["std", "vec", "Vec"]))
          .map(|_| {
            span_note_and_lint(cx, CLIPPY_BOX_VEC, ty.span,
                              "You seem to be trying to use Box<Vec<T>>. Did you mean to use Vec<T>?",
                              "Vec<T> is already on the heap, Box<Vec<T>> makes an extra allocation");
          });
        {
            // In case stuff gets moved around
            use collections::dlist::DList as DL1;
            use std::collections::dlist::DList as DL2;
            use std::collections::DList as DL3;
        }
        let dlists = [vec!["std","collections","dlist","DList"],
                      vec!["std","collections","DList"],
                      vec!["collections","dlist","DList"]];
        for path in dlists.iter() {
            if match_ty_unwrap(ty, path.as_slice()).is_some() {
                span_note_and_lint(cx, CLIPPY_DLIST, ty.span,
                                   "You seem to be trying to use a DList. Perhaps you meant some other data structure?",
                                   "A RingBuf might work.");
                return;
            }
        }
    }
}