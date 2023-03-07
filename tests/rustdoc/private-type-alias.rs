type MyResultPriv<T> = Result<T, u16>;
pub type MyResultPub<T> = Result<T, u64>;

// @has private_type_alias/fn.get_result_priv.html '//pre' 'Result<u8, u16>'
pub fn get_result_priv() -> MyResultPriv<u8> {
    panic!();
}

// @has private_type_alias/fn.get_result_pub.html '//pre' 'MyResultPub<u32>'
pub fn get_result_pub() -> MyResultPub<u32> {
    panic!();
}

pub type PubRecursive = u16;
type PrivRecursive3 = u8;
type PrivRecursive2 = PubRecursive;
type PrivRecursive1 = PrivRecursive3;

// PrivRecursive1 is expanded twice and stops at u8
// PrivRecursive2 is expanded once and stops at public type alias PubRecursive
// @has private_type_alias/fn.get_result_recursive.html '//pre' '(u8, PubRecursive)'
pub fn get_result_recursive() -> (PrivRecursive1, PrivRecursive2) {
    panic!();
}

type MyLifetimePriv<'a> = &'a isize;

// @has private_type_alias/fn.get_lifetime_priv.html '//pre' "&'static isize"
pub fn get_lifetime_priv() -> MyLifetimePriv<'static> {
    panic!();
}
