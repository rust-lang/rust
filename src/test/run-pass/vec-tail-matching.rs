struct Foo {
    string: ~str
}

pub fn main() {
    let x = [
        Foo { string: ~"foo" },
        Foo { string: ~"bar" },
        Foo { string: ~"baz" }
    ];
    match x {
        [first, ..tail] => {
            assert!(first.string == ~"foo");
            assert_eq!(tail.len(), 2);
            assert!(tail[0].string == ~"bar");
            assert!(tail[1].string == ~"baz");

            match tail {
                [Foo { _ }, _, Foo { _ }, ..tail] => {
                    ::std::util::unreachable();
                }
                [Foo { string: a }, Foo { string: b }] => {
                    assert_eq!(a, ~"bar");
                    assert_eq!(b, ~"baz");
                }
                _ => {
                    ::std::util::unreachable();
                }
            }
        }
        _ => {
            ::std::util::unreachable();
        }
    }
}
