pub enum Foo {
    //@ is "$.index[?(@.name=='Addition')].inner.variant.discriminant.value" '"0"'
    //@ is "$.index[?(@.name=='Addition')].inner.variant.discriminant.expr" '"{ _ }"'
    Addition = 0 + 0,
    //@ is "$.index[?(@.name=='Bin')].inner.variant.discriminant.value" '"1"'
    //@ is "$.index[?(@.name=='Bin')].inner.variant.discriminant.expr" '"0b1"'
    Bin = 0b1,
    //@ is "$.index[?(@.name=='Oct')].inner.variant.discriminant.value" '"2"'
    //@ is "$.index[?(@.name=='Oct')].inner.variant.discriminant.expr" '"0o2"'
    Oct = 0o2,
    //@ is "$.index[?(@.name=='PubConst')].inner.variant.discriminant.value" '"3"'
    //@ is "$.index[?(@.name=='PubConst')].inner.variant.discriminant.expr" '"THREE"'
    PubConst = THREE,
    //@ is "$.index[?(@.name=='Hex')].inner.variant.discriminant.value" '"4"'
    //@ is "$.index[?(@.name=='Hex')].inner.variant.discriminant.expr" '"0x4"'
    Hex = 0x4,
    //@ is "$.index[?(@.name=='Cast')].inner.variant.discriminant.value" '"5"'
    //@ is "$.index[?(@.name=='Cast')].inner.variant.discriminant.expr" '"{ _ }"'
    Cast = 5 as isize,
    //@ is "$.index[?(@.name=='PubCall')].inner.variant.discriminant.value" '"6"'
    //@ is "$.index[?(@.name=='PubCall')].inner.variant.discriminant.expr" '"{ _ }"'
    PubCall = six(),
    //@ is "$.index[?(@.name=='PrivCall')].inner.variant.discriminant.value" '"7"'
    //@ is "$.index[?(@.name=='PrivCall')].inner.variant.discriminant.expr" '"{ _ }"'
    PrivCall = seven(),
    //@ is "$.index[?(@.name=='PrivConst')].inner.variant.discriminant.value" '"8"'
    //@ is "$.index[?(@.name=='PrivConst')].inner.variant.discriminant.expr" '"EIGHT"'
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
