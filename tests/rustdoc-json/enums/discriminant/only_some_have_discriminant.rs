pub enum Foo {
    //@ jq .index[] | select(.name == "Has").inner.variant.discriminant? == {"expr":"0", "value":"0"}
    Has = 0,
    //@ jq .index[] | select(.name == "Doesnt").inner.variant.discriminant? == null
    Doesnt,
    //@ jq .index[] | select(.name == "AlsoDoesnt").inner.variant.discriminant? == null
    AlsoDoesnt,
    //@ jq .index[] | select(.name == "AlsoHas").inner.variant.discriminant? == {"expr":"44", "value":"44"}
    AlsoHas = 44,
}
