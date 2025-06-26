#[repr(i8)]
pub enum Ordering {
    //@ arg less .index[] | select(.name == "Less").inner.variant.discriminant?
    //@ jq $less.expr? == "-1"
    //@ jq $less.value? == "-1"
    Less = -1,
    //@ arg equal .index[] | select(.name == "Equal").inner.variant.discriminant?
    //@ jq $equal.expr? == "0"
    //@ jq $equal.value? == "0"
    Equal = 0,
    //@ arg greater .index[] | select(.name == "Greater").inner.variant.discriminant?
    //@ jq $greater.expr? == "1"
    //@ jq $greater.value? == "1"
    Greater = 1,
}
