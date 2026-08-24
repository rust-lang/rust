//@ run-pass

#![feature(specialization)]
#![allow(incomplete_features)]

// Tests that a `default impl` does not need all items, but does contribute to
// the chain of specialization.

// A partial `default impl` at each level of a 3-level chain.

trait Foo {
    type Assoc;
    const N: u32;
    fn from_root(&self) -> &'static str;
    fn from_mid(&self) -> &'static str;
    fn from_leaf(&self) -> &'static str;
    fn from_trait(&self) -> &'static str {
        "trait body"
    }
}

// root: assoc type, assoc const, one method
default impl<T> Foo for T {
    type Assoc = u8;
    const N: u32 = 1;
    fn from_root(&self) -> &'static str {
        "root"
    }
}

// middle: one method
default impl<T: Copy> Foo for T {
    fn from_mid(&self) -> &'static str {
        "mid"
    }
}

// leaf: one method. Everything else must come from the two ancestors, except
// `from_trait`, which no impl in the chain defines.
impl Foo for u32 {
    fn from_leaf(&self) -> &'static str {
        "leaf"
    }
}

// sibling leaf: overrides every inherited item, including assoc type and const
impl Foo for i8 {
    type Assoc = bool;
    const N: u32 = 2;
    fn from_root(&self) -> &'static str {
        "i8 root"
    }
    fn from_mid(&self) -> &'static str {
        "i8 mid"
    }
    fn from_leaf(&self) -> &'static str {
        "i8 leaf"
    }
    fn from_trait(&self) -> &'static str {
        "i8 trait"
    }
}

fn generic<T: Foo>(t: &T) -> [&'static str; 4] {
    [t.from_root(), t.from_mid(), t.from_leaf(), t.from_trait()]
}

// An empty `default impl`, and an empty real impl that inherits every item.

trait Marker {
    type A;
    fn m(&self) -> &'static str;
}

// Contributes nothing at all, and is still accepted.
default impl<T> Marker for T {}

// Covers every item of the trait.
default impl<T: Copy> Marker for T {
    type A = u8;
    fn m(&self) -> &'static str {
        "from default impl"
    }
}

// Declaration of intent and nothing else. This is what the `default impl` above
// is missing, and the only thing it is missing.
impl Marker for u32 {}

fn main() {
    // inherited across the chain, via a concrete receiver...
    assert_eq!(0u32.from_root(), "root");
    assert_eq!(0u32.from_mid(), "mid");
    assert_eq!(0u32.from_leaf(), "leaf");
    assert_eq!(0u32.from_trait(), "trait body");
    assert_eq!(<u32 as Foo>::N, 1);
    // The omitting impl finalizes the ancestor's definition, so this normalizes.
    let _: <u32 as Foo>::Assoc = 0u8;

    // ...and through a generic bound
    assert_eq!(generic(&0u32), ["root", "mid", "leaf", "trait body"]);
    assert_eq!(generic(&0i8), ["i8 root", "i8 mid", "i8 leaf", "i8 trait"]);
    assert_eq!(<i8 as Foo>::N, 2);
    let _: <i8 as Foo>::Assoc = true;

    // empty impl really does implement: method, projection, and vtable
    assert_eq!(0u32.m(), "from default impl");
    let _: <u32 as Marker>::A = 0u8;
    let _: &dyn Marker<A = u8> = &0u32;

}
