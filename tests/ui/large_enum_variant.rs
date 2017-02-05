#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(large_enum_variant)]

enum LargeEnum {
    A(i32),
    B([i32; 8000]), //~ ERROR large size difference between variants
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    //~| SUGGESTION Box<[i32; 8000]>
}

enum GenericEnumOk<T> {
    A(i32),
    B([T; 8000]),
}

enum GenericEnum2<T> {
    A(i32),
    B([i32; 8000]),
    C(T, [i32; 8000]), //~ ERROR large size difference between variants
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
}

trait SomeTrait {
    type Item;
}

enum LargeEnumGeneric<A: SomeTrait> {
    Var(A::Item), // regression test, this used to ICE
}

enum LargeEnum2 {
    VariantOk(i32, u32),
    ContainingLargeEnum(LargeEnum), //~ ERROR large size difference between variants
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    //~| SUGGESTION Box<LargeEnum>
}
enum LargeEnum3 {
    ContainingMoreThanOneField(i32, [i32; 8000], [i32; 9500]), //~ ERROR large size difference between variants
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
    VoidVariant,
    StructLikeLittle { x: i32, y: i32 },
}

enum LargeEnum4 {
    VariantOk(i32, u32),
    StructLikeLarge { x: [i32; 8000], y: i32 }, //~ ERROR large size difference between variants
    //~^ HELP consider boxing the large fields to reduce the total size of the enum
}

enum LargeEnum5 {
    VariantOk(i32, u32),
    StructLikeLarge2 { //~ ERROR large size difference between variants
        x:
        [i32; 8000] //~ SUGGESTION Box<[i32; 8000]>
        //~^ HELP consider boxing the large fields to reduce the total size of the enum
    },
}

enum LargeEnumOk {
    LargeA([i32; 8000]),
    LargeB([i32; 8001]),
}

fn main() {

}
