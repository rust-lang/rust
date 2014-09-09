// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(advanced_slice_patterns)]

fn match_vecs<'a, T>(l1: &'a [T], l2: &'a [T]) -> &'static str {
    match (l1, l2) {
        ([], []) => "both empty",
        ([], [..]) | ([..], []) => "one empty",
        ([..], [..]) => "both non-empty"
    }
}

fn match_vecs_cons<'a, T>(l1: &'a [T], l2: &'a [T]) -> &'static str {
    match (l1, l2) {
        ([], []) => "both empty",
        ([], [_, ..]) | ([_, ..], []) => "one empty",
        ([_, ..], [_, ..]) => "both non-empty"
    }
}

fn match_vecs_snoc<'a, T>(l1: &'a [T], l2: &'a [T]) -> &'static str {
    match (l1, l2) {
        ([], []) => "both empty",
        ([], [.., _]) | ([.., _], []) => "one empty",
        ([.., _], [.., _]) => "both non-empty"
    }
}

fn match_nested_vecs_cons<'a, T>(l1: Option<&'a [T]>, l2: Result<&'a [T], ()>) -> &'static str {
    match (l1, l2) {
        (Some([]), Ok([])) => "Some(empty), Ok(empty)",
        (Some([_, ..]), Ok(_)) | (Some([_, ..]), Err(())) => "Some(non-empty), any",
        (None, Ok([])) | (None, Err(())) | (None, Ok([_])) => "None, Ok(less than one element)",
        (None, Ok([_, _, ..])) => "None, Ok(at least two elements)",
        _ => "other"
    }
}

fn match_nested_vecs_snoc<'a, T>(l1: Option<&'a [T]>, l2: Result<&'a [T], ()>) -> &'static str {
    match (l1, l2) {
        (Some([]), Ok([])) => "Some(empty), Ok(empty)",
        (Some([.., _]), Ok(_)) | (Some([.., _]), Err(())) => "Some(non-empty), any",
        (None, Ok([])) | (None, Err(())) | (None, Ok([_])) => "None, Ok(less than one element)",
        (None, Ok([.., _, _])) => "None, Ok(at least two elements)",
        _ => "other"
    }
}

fn main() {
    assert_eq!(match_vecs(&[1i, 2], &[2i, 3]), "both non-empty");
    assert_eq!(match_vecs(&[], &[1i, 2, 3, 4]), "one empty");
    assert_eq!(match_vecs::<uint>(&[], &[]), "both empty");
    assert_eq!(match_vecs(&[1i, 2, 3], &[]), "one empty");

    assert_eq!(match_vecs_cons(&[1i, 2], &[2i, 3]), "both non-empty");
    assert_eq!(match_vecs_cons(&[], &[1i, 2, 3, 4]), "one empty");
    assert_eq!(match_vecs_cons::<uint>(&[], &[]), "both empty");
    assert_eq!(match_vecs_cons(&[1i, 2, 3], &[]), "one empty");

    assert_eq!(match_vecs_snoc(&[1i, 2], &[2i, 3]), "both non-empty");
    assert_eq!(match_vecs_snoc(&[], &[1i, 2, 3, 4]), "one empty");
    assert_eq!(match_vecs_snoc::<uint>(&[], &[]), "both empty");
    assert_eq!(match_vecs_snoc(&[1i, 2, 3], &[]), "one empty");

    assert_eq!(match_nested_vecs_cons(None, Ok::<&[_], ()>(&[4u, 2u])),
               "None, Ok(at least two elements)");
    assert_eq!(match_nested_vecs_cons::<uint>(None, Err(())), "None, Ok(less than one element)");
    assert_eq!(match_nested_vecs_cons::<bool>(Some::<&[_]>(&[]), Ok::<&[_], ()>(&[])),
               "Some(empty), Ok(empty)");
    assert_eq!(match_nested_vecs_cons(Some::<&[_]>(&[1i]), Err(())), "Some(non-empty), any");
    assert_eq!(match_nested_vecs_cons(Some::<&[_]>(&[(42i, ())]), Ok::<&[_], ()>(&[(1i, ())])),
               "Some(non-empty), any");

    assert_eq!(match_nested_vecs_snoc(None, Ok::<&[_], ()>(&[4u, 2u])),
               "None, Ok(at least two elements)");
    assert_eq!(match_nested_vecs_snoc::<uint>(None, Err(())), "None, Ok(less than one element)");
    assert_eq!(match_nested_vecs_snoc::<bool>(Some::<&[_]>(&[]), Ok::<&[_], ()>(&[])),
               "Some(empty), Ok(empty)");
    assert_eq!(match_nested_vecs_snoc(Some::<&[_]>(&[1i]), Err(())), "Some(non-empty), any");
    assert_eq!(match_nested_vecs_snoc(Some::<&[_]>(&[(42i, ())]), Ok::<&[_], ()>(&[(1i, ())])),
               "Some(non-empty), any");
}
