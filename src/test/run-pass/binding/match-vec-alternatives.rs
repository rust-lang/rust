// run-pass
#![feature(slice_patterns)]

fn match_vecs<'a, T>(l1: &'a [T], l2: &'a [T]) -> &'static str {
    match (l1, l2) {
        (&[], &[]) => "both empty",
        (&[], &[..]) | (&[..], &[]) => "one empty",
        (&[..], &[..]) => "both non-empty"
    }
}

fn match_vecs_cons<'a, T>(l1: &'a [T], l2: &'a [T]) -> &'static str {
    match (l1, l2) {
        (&[], &[]) => "both empty",
        (&[], &[_, ..]) | (&[_, ..], &[]) => "one empty",
        (&[_, ..], &[_, ..]) => "both non-empty"
    }
}

fn match_vecs_snoc<'a, T>(l1: &'a [T], l2: &'a [T]) -> &'static str {
    match (l1, l2) {
        (&[], &[]) => "both empty",
        (&[], &[.., _]) | (&[.., _], &[]) => "one empty",
        (&[.., _], &[.., _]) => "both non-empty"
    }
}

fn match_nested_vecs_cons<'a, T>(l1: Option<&'a [T]>, l2: Result<&'a [T], ()>) -> &'static str {
    match (l1, l2) {
        (Some(&[]), Ok(&[])) => "Some(empty), Ok(empty)",
        (Some(&[_, ..]), Ok(_)) | (Some(&[_, ..]), Err(())) => "Some(non-empty), any",
        (None, Ok(&[])) | (None, Err(())) | (None, Ok(&[_])) => "None, Ok(less than one element)",
        (None, Ok(&[_, _, ..])) => "None, Ok(at least two elements)",
        _ => "other"
    }
}

fn match_nested_vecs_snoc<'a, T>(l1: Option<&'a [T]>, l2: Result<&'a [T], ()>) -> &'static str {
    match (l1, l2) {
        (Some(&[]), Ok(&[])) => "Some(empty), Ok(empty)",
        (Some(&[.., _]), Ok(_)) | (Some(&[.., _]), Err(())) => "Some(non-empty), any",
        (None, Ok(&[])) | (None, Err(())) | (None, Ok(&[_])) => "None, Ok(less than one element)",
        (None, Ok(&[.., _, _])) => "None, Ok(at least two elements)",
        _ => "other"
    }
}

fn main() {
    assert_eq!(match_vecs(&[1, 2], &[2, 3]), "both non-empty");
    assert_eq!(match_vecs(&[], &[1, 2, 3, 4]), "one empty");
    assert_eq!(match_vecs::<usize>(&[], &[]), "both empty");
    assert_eq!(match_vecs(&[1, 2, 3], &[]), "one empty");

    assert_eq!(match_vecs_cons(&[1, 2], &[2, 3]), "both non-empty");
    assert_eq!(match_vecs_cons(&[], &[1, 2, 3, 4]), "one empty");
    assert_eq!(match_vecs_cons::<usize>(&[], &[]), "both empty");
    assert_eq!(match_vecs_cons(&[1, 2, 3], &[]), "one empty");

    assert_eq!(match_vecs_snoc(&[1, 2], &[2, 3]), "both non-empty");
    assert_eq!(match_vecs_snoc(&[], &[1, 2, 3, 4]), "one empty");
    assert_eq!(match_vecs_snoc::<usize>(&[], &[]), "both empty");
    assert_eq!(match_vecs_snoc(&[1, 2, 3], &[]), "one empty");

    assert_eq!(match_nested_vecs_cons(None, Ok::<&[_], ()>(&[4_usize, 2_usize])),
               "None, Ok(at least two elements)");
    assert_eq!(match_nested_vecs_cons::<usize>(None, Err(())), "None, Ok(less than one element)");
    assert_eq!(match_nested_vecs_cons::<bool>(Some::<&[_]>(&[]), Ok::<&[_], ()>(&[])),
               "Some(empty), Ok(empty)");
    assert_eq!(match_nested_vecs_cons(Some::<&[_]>(&[1]), Err(())), "Some(non-empty), any");
    assert_eq!(match_nested_vecs_cons(Some::<&[_]>(&[(42, ())]), Ok::<&[_], ()>(&[(1, ())])),
               "Some(non-empty), any");

    assert_eq!(match_nested_vecs_snoc(None, Ok::<&[_], ()>(&[4_usize, 2_usize])),
               "None, Ok(at least two elements)");
    assert_eq!(match_nested_vecs_snoc::<usize>(None, Err(())), "None, Ok(less than one element)");
    assert_eq!(match_nested_vecs_snoc::<bool>(Some::<&[_]>(&[]), Ok::<&[_], ()>(&[])),
               "Some(empty), Ok(empty)");
    assert_eq!(match_nested_vecs_snoc(Some::<&[_]>(&[1]), Err(())), "Some(non-empty), any");
    assert_eq!(match_nested_vecs_snoc(Some::<&[_]>(&[(42, ())]), Ok::<&[_], ()>(&[(1, ())])),
               "Some(non-empty), any");
}
