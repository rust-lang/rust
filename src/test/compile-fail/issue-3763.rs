// xfail-test
mod my_mod {
    pub struct MyStruct {
        priv priv_field: int
    }
    pub fn MyStruct () -> MyStruct {
        MyStruct {priv_field: 4}
    }
}

fn main() {
    let my_struct = my_mod::MyStruct();
    let _woohoo = (&my_struct).priv_field; // compiles but shouldn't
    let _woohoo = (~my_struct).priv_field; // ditto
    let _woohoo = (@my_struct).priv_field; // ditto
   // let nope = my_struct.priv_field;       // compile error as expected
}
