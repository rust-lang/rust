// Test that we do not permit moves from &[] matched by a vec pattern.

#![feature(slice_patterns)]

#[derive(Clone, Debug)]
struct Foo {
    string: String
}

pub fn main() {
    let x = vec![
        Foo { string: "foo".to_string() },
        Foo { string: "bar".to_string() },
        Foo { string: "baz".to_string() }
    ];
    let x: &[Foo] = &x;
    match *x {
        [_, ref tail..] => {
            match tail {
            //~^ ERROR cannot move out of type `[Foo]`
                &[Foo { string: a },
                  Foo { string: b }] => {
                }
                _ => {
                    unreachable!();
                }
            }
            let z = tail[0].clone();
            println!("{:?}", z);
        }
        _ => {
            unreachable!();
        }
    }
}
