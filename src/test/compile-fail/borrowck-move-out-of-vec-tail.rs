// Test that we do not permit moves from &[] matched by a vec pattern.

#[deriving(Clone)]
struct Foo {
    string: ~str
}

pub fn main() {
    let x = ~[
        Foo { string: ~"foo" },
        Foo { string: ~"bar" },
        Foo { string: ~"baz" }
    ];
    match x {
        [_, ..tail] => {
            match tail {
                [Foo { string: a }, Foo { string: b }] => {
                    //~^ ERROR cannot move out of dereference of & pointer
                    //~^^ ERROR cannot move out of dereference of & pointer
                }
                _ => {
                    unreachable!();
                }
            }
            let z = tail[0].clone();
            info2!("{:?}", z);
        }
        _ => {
            unreachable!();
        }
    }
}
