pub enum Foo {
    //@ arg addition .index[] | select(.name == "Addition").inner.variant.discriminant?
    //@ jq $addition.value? == "0"
    //@ jq $addition.expr? == "{ _ }"
    Addition = 0 + 0,
    //@ arg bin .index[] | select(.name == "Bin").inner.variant.discriminant?
    //@ jq $bin.value? == "1"
    //@ jq $bin.expr? == "0b1"
    Bin = 0b1,
    //@ arg oct .index[] | select(.name == "Oct").inner.variant.discriminant?
    //@ jq $oct.value? == "2"
    //@ jq $oct.expr? == "0o2"
    Oct = 0o2,
    //@ arg pub_const .index[] | select(.name == "PubConst").inner.variant.discriminant?
    //@ jq $pub_const.value? == "3"
    //@ jq $pub_const.expr? == "THREE"
    PubConst = THREE,
    //@ arg hex .index[] | select(.name == "Hex").inner.variant.discriminant?
    //@ jq $hex.value? == "4"
    //@ jq $hex.expr? == "0x4"
    Hex = 0x4,
    //@ arg cast .index[] | select(.name == "Cast").inner.variant.discriminant?
    //@ jq $cast.value? == "5"
    //@ jq $cast.expr? == "{ _ }"
    Cast = 5 as isize,
    //@ arg pub_call .index[] | select(.name == "PubCall").inner.variant.discriminant?
    //@ jq $pub_call.value? == "6"
    //@ jq $pub_call.expr? == "{ _ }"
    PubCall = six(),
    //@ arg priv_call .index[] | select(.name == "PrivCall").inner.variant.discriminant?
    //@ jq $priv_call.value? == "7"
    //@ jq $priv_call.expr? == "{ _ }"
    PrivCall = seven(),
    //@ arg priv_const .index[] | select(.name == "PrivConst").inner.variant.discriminant?
    //@ jq $priv_const.value? == "8"
    //@ jq $priv_const.expr? == "EIGHT"
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
