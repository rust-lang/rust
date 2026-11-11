use crate::config::native_libs::{NativeLibParts, split_native_lib_value};

#[test]
fn split() {
    // This is a unit test for some implementation details, so consider deleting
    // it if it gets in the way.
    use NativeLibParts as P;

    let examples = &[
        ("", P { kind: None, modifiers: None, name: "", new_name: None }),
        ("foo", P { kind: None, modifiers: None, name: "foo", new_name: None }),
        ("foo:", P { kind: None, modifiers: None, name: "foo", new_name: Some("") }),
        ("foo:bar", P { kind: None, modifiers: None, name: "foo", new_name: Some("bar") }),
        (":bar", P { kind: None, modifiers: None, name: "", new_name: Some("bar") }),
        ("kind=foo", P { kind: Some("kind"), modifiers: None, name: "foo", new_name: None }),
        (":mods=foo", P { kind: Some(""), modifiers: Some("mods"), name: "foo", new_name: None }),
        (
            ":mods=:bar",
            P { kind: Some(""), modifiers: Some("mods"), name: "", new_name: Some("bar") },
        ),
        (
            "kind=foo:bar",
            P { kind: Some("kind"), modifiers: None, name: "foo", new_name: Some("bar") },
        ),
        (
            "kind:mods=foo",
            P { kind: Some("kind"), modifiers: Some("mods"), name: "foo", new_name: None },
        ),
        (
            "kind:mods=foo:bar",
            P { kind: Some("kind"), modifiers: Some("mods"), name: "foo", new_name: Some("bar") },
        ),
        ("::==::", P { kind: Some(""), modifiers: Some(":"), name: "=", new_name: Some(":") }),
        ("==::==", P { kind: Some(""), modifiers: None, name: "=", new_name: Some(":==") }),
    ];

    for &(value, ref expected) in examples {
        println!("{value:?}");
        let actual = split_native_lib_value(value);
        assert_eq!(&actual, expected);
    }
}
