//@ run-pass

#![allow(dead_code)]

#[derive(Debug)]
enum MyEnum {
    Variant1(Vec<u8>),
    Variant2,
    Variant3,
    Variant4,
}

fn f(arg1: &bool, arg2: &bool, arg3: bool) -> MyStruct {
    if *arg1 {
        println!("{:?}", f(&arg2, arg2, arg3));
        MyStruct(None)
    } else {
        match if arg3 { Some(MyEnum::Variant3) } else { None } {
            Some(t) => {
                let ah = t;
                return MyStruct(Some(ah));
            }
            _ => MyStruct(None)
        }
    }
}

#[derive(Debug)]
struct MyStruct(Option<MyEnum>);

fn main() {
    let arg1 = true;
    let arg2 = false;
    f(&arg1, &arg2, true);
}
