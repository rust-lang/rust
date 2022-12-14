pub enum Foo {
    // @is "$.index[*][?(@.name=='Addition')].inner.variant_inner.value" '"0"'
    // @is "$.index[*][?(@.name=='Addition')].inner.variant_inner.expr" '"{ _ }"'
    Addition = 0 + 0,
    // @is "$.index[*][?(@.name=='Bin')].inner.variant_inner.value" '"1"'
    // @is "$.index[*][?(@.name=='Bin')].inner.variant_inner.expr" '"0b1"'
    Bin = 0b1,
    // @is "$.index[*][?(@.name=='Oct')].inner.variant_inner.value" '"2"'
    // @is "$.index[*][?(@.name=='Oct')].inner.variant_inner.expr" '"0o2"'
    Oct = 0o2,
    // @is "$.index[*][?(@.name=='PubConst')].inner.variant_inner.value" '"3"'
    // @is "$.index[*][?(@.name=='PubConst')].inner.variant_inner.expr" '"THREE"'
    PubConst = THREE,
    // @is "$.index[*][?(@.name=='Hex')].inner.variant_inner.value" '"4"'
    // @is "$.index[*][?(@.name=='Hex')].inner.variant_inner.expr" '"0x4"'
    Hex = 0x4,
    // @is "$.index[*][?(@.name=='Cast')].inner.variant_inner.value" '"5"'
    // @is "$.index[*][?(@.name=='Cast')].inner.variant_inner.expr" '"{ _ }"'
    Cast = 5 as isize,
    // @is "$.index[*][?(@.name=='PubCall')].inner.variant_inner.value" '"6"'
    // @is "$.index[*][?(@.name=='PubCall')].inner.variant_inner.expr" '"{ _ }"'
    PubCall = six(),
    // @is "$.index[*][?(@.name=='PrivCall')].inner.variant_inner.value" '"7"'
    // @is "$.index[*][?(@.name=='PrivCall')].inner.variant_inner.expr" '"{ _ }"'
    PrivCall = seven(),
    // @is "$.index[*][?(@.name=='PrivConst')].inner.variant_inner.value" '"8"'
    // @is "$.index[*][?(@.name=='PrivConst')].inner.variant_inner.expr" '"EIGHT"'
    PrivConst = EIGHT,
}

pub const THREE: isize = 3;
const EIGHT: isize = 8;

pub const fn six() -> isize {
    6
}
const fn seven() -> isize {
    7
}
