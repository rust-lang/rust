pub enum Foo {
    // @is "$.index[*][?(@.name=='Addition')].inner.discriminant.value" '"0"'
    // @is "$.index[*][?(@.name=='Addition')].inner.discriminant.expr" '"{ _ }"'
    Addition = 0 + 0,
    // @is "$.index[*][?(@.name=='Bin')].inner.discriminant.value" '"1"'
    // @is "$.index[*][?(@.name=='Bin')].inner.discriminant.expr" '"0b1"'
    Bin = 0b1,
    // @is "$.index[*][?(@.name=='Oct')].inner.discriminant.value" '"2"'
    // @is "$.index[*][?(@.name=='Oct')].inner.discriminant.expr" '"0o2"'
    Oct = 0o2,
    // @is "$.index[*][?(@.name=='PubConst')].inner.discriminant.value" '"3"'
    // @is "$.index[*][?(@.name=='PubConst')].inner.discriminant.expr" '"THREE"'
    PubConst = THREE,
    // @is "$.index[*][?(@.name=='Hex')].inner.discriminant.value" '"4"'
    // @is "$.index[*][?(@.name=='Hex')].inner.discriminant.expr" '"0x4"'
    Hex = 0x4,
    // @is "$.index[*][?(@.name=='Cast')].inner.discriminant.value" '"5"'
    // @is "$.index[*][?(@.name=='Cast')].inner.discriminant.expr" '"{ _ }"'
    Cast = 5 as isize,
    // @is "$.index[*][?(@.name=='PubCall')].inner.discriminant.value" '"6"'
    // @is "$.index[*][?(@.name=='PubCall')].inner.discriminant.expr" '"{ _ }"'
    PubCall = six(),
    // @is "$.index[*][?(@.name=='PrivCall')].inner.discriminant.value" '"7"'
    // @is "$.index[*][?(@.name=='PrivCall')].inner.discriminant.expr" '"{ _ }"'
    PrivCall = seven(),
    // @is "$.index[*][?(@.name=='PrivConst')].inner.discriminant.value" '"8"'
    // @is "$.index[*][?(@.name=='PrivConst')].inner.discriminant.expr" '"EIGHT"'
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
