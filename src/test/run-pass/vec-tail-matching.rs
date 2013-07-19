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
        [ref first, ..tail] => {
            assert!(first.string == ~"foo");
            assert_eq!(tail.len(), 2);
            assert!(tail[0].string == ~"bar");
            assert!(tail[1].string == ~"baz");

            match tail {
                [Foo { _ }, _, Foo { _ }, ..tail] => {
                    ::std::util::unreachable();
                }
                [Foo { string: ref a }, Foo { string: ref b }] => {
                    assert_eq!("bar", a.slice(0, a.len()));
                    assert_eq!("baz", b.slice(0, b.len()));
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
