// pp-exact

#![feature(rustc_attrs)]

macro_rules! mac { ($($tt : tt) *) => () }

mac! {
    struct S { field1 : u8, field2 : u16, } impl Clone for S
    {
        fn clone() -> S
        {
            panic! () ;

        }
    }
}

mac! {
    a(aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
    aaaaaaaa aaaaaaaa) a
    [aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
    aaaaaaaa aaaaaaaa] a
    {
        aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
        aaaaaaaa aaaaaaaa aaaaaaaa
    } a
}

mac!(aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
aaaaaaaa aaaaaaaa);
mac![aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
aaaaaaaa aaaaaaaa];
mac! {
    aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
    aaaaaaaa aaaaaaaa
}

#[rustc_dummy(aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
aaaaaaaa aaaaaaaa aaaaaaaa)]
#[rustc_dummy[aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
aaaaaaaa aaaaaaaa aaaaaaaa]]
#[rustc_dummy {
    aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa
    aaaaaaaa aaaaaaaa
}]
#[rustc_dummy =
"aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa aaaaaaaa"]
fn main() {}
