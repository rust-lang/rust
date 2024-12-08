//@ check-pass

enum FooEnum {
    #[rustfmt::skip]
    Bar(i32),
}

struct FooStruct {
    #[rustfmt::skip]
    bar: i32,
}

fn main() {
    let foo_enum_bar = FooEnum::Bar(1);
    match foo_enum_bar {
        FooEnum::Bar(x) => {}
        _ => {}
    }

    let foo_struct = FooStruct { bar: 1 };
    match foo_struct {
        FooStruct {
            #[rustfmt::skip] bar
        } => {}
    }

    match 1 {
        0 => {}
        #[rustfmt::skip]
        _ => {}
    }

    let _another_foo_strunct = FooStruct {
        #[rustfmt::skip]
        bar: 1,
    };
}
