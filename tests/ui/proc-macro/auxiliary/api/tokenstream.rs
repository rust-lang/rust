use proc_macro::*;

fn assert_eq(l: TokenStream, r: TokenStream) {
    assert_eq!(l.to_string(), r.to_string());
    for (lt, rt) in l.into_iter().zip(r) {
        assert_eq!(lt.to_string(), rt.to_string());
    }
}

pub fn test() {
    assert_eq(TokenStream::new(), TokenStream::new());
    let mut stream = TokenStream::new();
    assert!(stream.is_empty());
    stream.extend(TokenStream::new());
    assert_eq(stream.clone(), TokenStream::new());

    let old = stream.clone();
    stream.extend(vec![TokenTree::Ident(Ident::new("foo", Span::call_site()))]);
    assert!(!stream.is_empty());
    assert!(old.is_empty());

    let stream2 = stream
        .clone()
        .into_iter()
        .inspect(|tree| assert_eq!(tree.to_string(), "foo"))
        .collect::<TokenStream>();
    assert_eq(stream.clone(), stream2);
}
