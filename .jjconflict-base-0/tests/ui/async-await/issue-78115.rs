// Regression test for issue #78115: "ICE: variable should be placed in scope earlier"

//@ check-pass
//@ edition:2018

#[allow(dead_code)]
struct Foo {
    a: ()
}

async fn _bar() {
    let foo = Foo { a: () };
    match foo {
        Foo { a: _a } | Foo { a: _a } if true => {}
        _ => {}
    }
}

fn main() {}
