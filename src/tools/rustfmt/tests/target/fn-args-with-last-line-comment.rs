// #1587
pub trait X {
    fn a(&self) -> &'static str;
    fn bcd(
        &self,
        c: &str,         // comment on this arg
        d: u16,          // comment on this arg
        e: &Vec<String>, // comment on this arg
    ) -> Box<Q>;
}

// #1595
fn foo(
    arg1: LongTypeName,
    arg2: LongTypeName,
    arg3: LongTypeName,
    arg4: LongTypeName,
    arg5: LongTypeName,
    arg6: LongTypeName,
    arg7: LongTypeName,
    //arg8: LongTypeName,
) {
    // do stuff
}
