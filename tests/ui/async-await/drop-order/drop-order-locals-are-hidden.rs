//@ edition:2018

async fn foobar_async(x: u32, (a, _, _c): (u32, u32, u32), _: u32, _y: u32) {
    assert_eq!(__arg1, (1, 2, 3)); //~ ERROR cannot find value `__arg1` in this scope [E0425]
    assert_eq!(__arg2, 4); //~ ERROR cannot find value `__arg2` in this scope [E0425]
}

async fn baz_async(ref mut x: u32, ref y: u32) {
    assert_eq!(__arg0, 1); //~ ERROR cannot find value `__arg0` in this scope [E0425]
    assert_eq!(__arg1, 2); //~ ERROR cannot find value `__arg1` in this scope [E0425]
}

fn main() {}
