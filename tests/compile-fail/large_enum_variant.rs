#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(large_enum_variant)]

enum LargeEnum {
    A(i32),
    B([i32; 8000]), //~ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    //~| SUGGESTION Box<[i32; 8000]>
}

enum GenericEnum<T> {
    A(i32),
    B([i32; 8000]), //~ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    //~| SUGGESTION Box<[i32; 8000]>
    C([T; 8000]),
    D(T, [i32; 8000]), //~ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
}

trait SomeTrait {
    type Item;
}

enum LargeEnumGeneric<A: SomeTrait> {
    Var(A::Item), // regression test, this used to ICE
}

enum AnotherLargeEnum {
    VariantOk(i32, u32),
    ContainingLargeEnum(LargeEnum), //~ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    //~| SUGGESTION Box<LargeEnum>
    ContainingMoreThanOneField(i32, [i32; 8000], [i32; 9500]), //~ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    VoidVariant,
    StructLikeLittle { x: i32, y: i32 },
    StructLikeLarge { x: [i32; 8000], y: i32 }, //~ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    StructLikeLarge2 {
        x:
        [i32; 8000] //~ SUGGESTION Box<[i32; 8000]>
    },
    //~^ ERROR large enum variant found
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
}

fn main() {

}
