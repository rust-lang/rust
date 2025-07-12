pub enum Foo {
    Variant {
        #[doc(hidden)]
        a: i32,
        b: i32,
        #[doc(hidden)]
        x: i32,
        y: i32,
    },
    //@ arg variant .index[] | select(.name == "Variant").inner.variant.kind?.struct
    //@ jq $variant.has_stripped_fields? == true
    //@ jq [$variant.fields[]?] == [.index[] | select(.name == "b" or .name == "y").id]
}
