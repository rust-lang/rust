enum FooEnum {
    #[test]
    //~^ ERROR expected an inert attribute, found an attribute macro
    Bar(i32),
}

struct FooStruct {
    #[test]
    //~^ ERROR expected an inert attribute, found an attribute macro
    bar: i32,
}

fn main() {
    let foo_enum_bar = FooEnum::Bar(1);
    match foo_enum_bar {
        FooEnum::Bar(x) => {},
        _ => {}
    }

    let foo_struct = FooStruct { bar: 1 };
    match foo_struct {
        FooStruct {
            #[test] bar
            //~^ ERROR expected an inert attribute, found an attribute macro
        } => {}
    }

    match 1 {
        0 => {}
        #[test]
        //~^ ERROR expected an inert attribute, found an attribute macro
        _ => {}
    }

    let _another_foo_strunct = FooStruct {
        #[test]
        //~^ ERROR expected an inert attribute, found an attribute macro
        bar: 1,
    };
}
