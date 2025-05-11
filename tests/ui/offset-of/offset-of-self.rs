use std::mem::offset_of;

struct C<T> {
    v: T,
    w: T,
}

struct S {
    v: u8,
    w: u16,
}

impl S {
    fn v_offs() -> usize {
        offset_of!(Self, v)
    }
    fn v_offs_wrong_syntax() {
        offset_of!(Self, Self::v); //~ ERROR offset_of expects dot-separated field and variant names
        offset_of!(S, Self); //~ ERROR no field `Self` on type `S`
    }
    fn offs_in_c() -> usize {
        offset_of!(C<Self>, w)
    }
    fn offs_in_c_colon() -> usize {
        offset_of!(C::<Self>, w)
    }
}

mod m {
    use std::mem::offset_of;
    fn off() {
        offset_of!(self::S, v); //~ ERROR cannot find type `S` in module
        offset_of!(super::S, v);
        offset_of!(crate::S, v);
    }
    impl super::n::T {
        fn v_offs_self() -> usize {
            offset_of!(Self, v) //~ ERROR field `v` of struct `T` is private
        }
    }
}

mod n {
    pub struct T { v: u8, }
}

fn main() {
    offset_of!(self::S, v);
    offset_of!(Self, v); //~ ERROR cannot find type `Self` in this scope

    offset_of!(S, self); //~ ERROR no field `self` on type `S`
    offset_of!(S, v.self); //~ ERROR no field `self` on type `u8`
}
