#![feature(plugin)]
#![plugin(clippy)]

#![allow(dead_code)]
#![allow(unused_variables)]
#![deny(large_enum_variant)]

enum LargeEnum {
    A(i32),
    B([i32; 8000]), //~ ERROR large enum variant found on variant `B`
}

enum GenericEnum<T> {
    A(i32),
    B([i32; 8000]), //~ ERROR large enum variant found on variant `B`
    C([T; 8000]),
    D(T, [i32; 8000]), //~ ERROR large enum variant found on variant `D`
}

trait SomeTrait {
    type Item;
}

enum LargeEnumGeneric<A: SomeTrait> {
    Var(A::Item), // regression test, this used to ICE
}

enum AnotherLargeEnum {
    VariantOk(i32, u32),
    ContainingLargeEnum(LargeEnum), //~ ERROR large enum variant found on variant `ContainingLargeEnum`
    ContainingMoreThanOneField(i32, [i32; 8000], [i32; 9500]), //~ ERROR large enum variant found on variant `ContainingMoreThanOneField`
    VoidVariant,
    StructLikeLittle { x: i32, y: i32 },
    StructLikeLarge { x: [i32; 8000], y: i32 }, //~ ERROR large enum variant found on variant `StructLikeLarge`
}

fn main() {

}
