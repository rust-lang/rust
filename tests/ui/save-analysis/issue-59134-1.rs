// compile-flags: -Zsave-analysis

// Check that this doesn't ICE when processing associated const (type).

fn func() {
    trait Trait {
        type MyType;
        const CONST: Self::MyType = bogus.field; //~ ERROR cannot find value `bogus`
    }
}

fn main() {}
