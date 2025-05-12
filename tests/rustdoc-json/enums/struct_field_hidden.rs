pub enum Foo {
    Variant {
        #[doc(hidden)]
        a: i32,
        //@ set b = "$.index[?(@.name=='b')].id"
        b: i32,
        #[doc(hidden)]
        x: i32,
        //@ set y = "$.index[?(@.name=='y')].id"
        y: i32,
    },
    //@ is "$.index[?(@.name=='Variant')].inner.variant.kind.struct.has_stripped_fields" true
    //@ is "$.index[?(@.name=='Variant')].inner.variant.kind.struct.fields[0]" $b
    //@ is "$.index[?(@.name=='Variant')].inner.variant.kind.struct.fields[1]" $y
    //@ count "$.index[?(@.name=='Variant')].inner.variant.kind.struct.fields[*]" 2
}
