// Issue 52985: Cause cycle error if user code provides no use case that allows an existential type
// to be inferred to a concrete type. This results in an infinite cycle during type normalization.

#![feature(existential_type)]

existential type Foo: Copy; //~ cycle detected

// make compiler happy about using 'Foo'
fn bar(x: Foo) -> Foo { x }

fn main() {
    let _: Foo = std::mem::transmute(0u8);
}
