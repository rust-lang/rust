// Issue 52985: user code provides no use case that allows an existential type
// We now emit a 'could not find defining uses' error

#![feature(existential_type)]

existential type Foo: Copy; //~ could not find defining uses

// make compiler happy about using 'Foo'
fn bar(x: Foo) -> Foo { x }

fn main() {
    let _: Foo = std::mem::transmute(0u8);
}
