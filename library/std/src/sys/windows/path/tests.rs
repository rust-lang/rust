use super::*;

#[test]
fn test_get_first_two_components() {
    assert_eq!(
        get_first_two_components(br"server\share", is_verbatim_sep),
        Some((&b"server"[..], &b"share"[..])),
    );

    assert_eq!(
        get_first_two_components(br"server\", is_verbatim_sep),
        Some((&b"server"[..], &b""[..]))
    );

    assert_eq!(
        get_first_two_components(br"\server\", is_verbatim_sep),
        Some((&b""[..], &b"server"[..]))
    );

    assert_eq!(get_first_two_components(br"there are no separators here", is_verbatim_sep), None,);
}
