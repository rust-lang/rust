struct Foo(String);
struct Bar { baz: String }

fn foo(foo: Foo) -> bool {
    match foo {
        Foo("hi".to_owned()) => true,
        //~^ error: expected a pattern, found a method call
        _ => false
    }
}

fn bar(bar: Bar) -> bool {
    match bar {
        Bar { baz: "hi".to_owned() } => true,
        //~^ error: expected a pattern, found a method call
        _ => false
    }
}

fn baz() { // issue #90121
    let foo = vec!["foo".to_string()];

    match foo.as_slice() {
        &["foo".to_string()] => {}
        //~^ error: expected a pattern, found a method call
        _ => {}
    };
}

fn main() {
    if let (-1.some(4)) = (0, Some(4)) {}
    //~^ error: expected a pattern, found a method call

    if let (-1.Some(4)) = (0, Some(4)) {}
    //~^ error: expected one of `)`, `,`, `...`, `..=`, `..`, or `|`, found `.`
    //~| help: missing `,`
}
