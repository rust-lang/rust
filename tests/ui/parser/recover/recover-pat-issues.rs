struct Foo(String);
struct Bar { baz: String }

fn foo(foo: Foo) -> bool {
    match foo {
        Foo("hi".to_owned()) => true,
        //~^ error: expected a pattern, found an expression
        _ => false
    }
}

fn bar(bar: Bar) -> bool {
    match bar {
        Bar { baz: "hi".to_owned() } => true,
        //~^ error: expected a pattern, found an expression
        _ => false
    }
}

/// Issue #90121
fn baz() {
    let foo = vec!["foo".to_string()];

    match foo.as_slice() {
        &["foo".to_string()] => {}
        //~^ error: expected a pattern, found an expression
        _ => {}
    };
}

/// Issue #104996
fn qux() {
    struct Magic(pub u16);
    const MAGIC: Magic = Magic(42);

    if let Some(MAGIC.0 as usize) = None::<usize> {}
    //~^ error: expected a pattern, found an expression
}

fn main() {
    if let (-1.some(4)) = (0, Some(4)) {}
    //~^ error: expected a pattern, found an expression

    if let (-1.Some(4)) = (0, Some(4)) {}
    //~^ error: expected a pattern, found an expression
}
