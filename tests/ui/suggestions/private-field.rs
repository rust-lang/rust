// compile-flags: --crate-type lib
pub struct S {
    pub val: string::MyString,
}

pub fn test(s: S) {
    dbg!(s.cap) //~ ERROR: no field `cap` on type `S` [E0609]
}

pub(crate) mod string {

    pub struct MyString {
        buf: MyVec,
    }

    struct MyVec {
        cap: usize,
    }
}
