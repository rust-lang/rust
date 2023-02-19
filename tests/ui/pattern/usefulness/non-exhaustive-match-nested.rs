enum T { A(U), B }
enum U { C, D }

fn match_nested_vecs<'a, T>(l1: Option<&'a [T]>, l2: Result<&'a [T], ()>) -> &'static str {
    match (l1, l2) { //~ ERROR match is non-exhaustive
        (Some(&[]), Ok(&[])) => "Some(empty), Ok(empty)",
        (Some(&[_, ..]), Ok(_)) | (Some(&[_, ..]), Err(())) => "Some(non-empty), any",
        (None, Ok(&[])) | (None, Err(())) | (None, Ok(&[_])) => "None, Ok(less than one element)",
        (None, Ok(&[_, _, ..])) => "None, Ok(at least two elements)"
    }
}

fn main() {
    let x = T::A(U::C);
    match x { //~ ERROR match is non-exhaustive
        T::A(U::D) => { panic!("hello"); }
        T::B => { panic!("goodbye"); }
    }
}
