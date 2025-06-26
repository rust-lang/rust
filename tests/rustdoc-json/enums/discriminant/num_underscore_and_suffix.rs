#[repr(u32)]
pub enum Foo {
    //@ arg basic .index[] | select(.name == "Basic").inner.variant.discriminant?
    //@ jq $basic.value? == "0"
    //@ jq $basic.expr? == "0"
    Basic = 0,
    //@ arg suffix .index[] | select(.name == "Suffix").inner.variant.discriminant?
    //@ jq $suffix.value? == "10"
    //@ jq $suffix.expr? == "10u32"
    Suffix = 10u32,
    //@ arg underscore .index[] | select(.name == "Underscore").inner.variant.discriminant?
    //@ jq $underscore.value? == "100"
    //@ jq $underscore.expr? == "1_0_0"
    Underscore = 1_0_0,
    //@ arg suffix_underscore .index[] | select(.name == "SuffixUnderscore").inner.variant.discriminant?
    //@ jq $suffix_underscore.value? == "1000"
    //@ jq $suffix_underscore.expr? == "1_0_0_0u32"
    SuffixUnderscore = 1_0_0_0u32,
}
