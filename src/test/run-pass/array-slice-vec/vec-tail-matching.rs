// run-pass

#![feature(slice_patterns)]

struct Foo {
    string: &'static str
}

pub fn main() {
    let x = [
        Foo { string: "foo" },
        Foo { string: "bar" },
        Foo { string: "baz" }
    ];
    match x {
        [ref first, ref tail..] => {
            assert_eq!(first.string, "foo");
            assert_eq!(tail.len(), 2);
            assert_eq!(tail[0].string, "bar");
            assert_eq!(tail[1].string, "baz");

            match *(tail as &[_]) {
                [Foo { .. }, _, Foo { .. }, ref _tail..] => {
                    unreachable!();
                }
                [Foo { string: ref a }, Foo { string: ref b }] => {
                    assert_eq!("bar", &a[0..a.len()]);
                    assert_eq!("baz", &b[0..b.len()]);
                }
                _ => {
                    unreachable!();
                }
            }
        }
    }
}
