mod my_mod {
    pub struct MyStruct {
        priv_field: isize
    }
    pub fn MyStruct () -> MyStruct {
        MyStruct {priv_field: 4}
    }
    impl MyStruct {
        fn happyfun(&self) {}
    }
}

fn main() {
    let my_struct = my_mod::MyStruct();
    let _woohoo = (&my_struct).priv_field;
    //~^ ERROR field `priv_field` of struct `my_mod::MyStruct` is private

    let _woohoo = (Box::new(my_struct)).priv_field;
    //~^ ERROR field `priv_field` of struct `my_mod::MyStruct` is private

    (&my_struct).happyfun();               //~ ERROR method `happyfun` is private

    (Box::new(my_struct)).happyfun();          //~ ERROR method `happyfun` is private
    let nope = my_struct.priv_field;
    //~^ ERROR field `priv_field` of struct `my_mod::MyStruct` is private
}
