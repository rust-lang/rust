//! regression test for #128347
#![crate_name = "foo"]

//@ snapshot perms foo/enum.Permissions.html '//pre/code'

#[repr(u8)]
pub enum Permissions {
    Guest = b'%',
    User = b'$',
    System = b'@',
    Absolute = b'#',
}
