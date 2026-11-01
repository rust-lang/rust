fn main() {
    let value = 1;

    match SomeStruct(value) {
        StructConst1(_) => { },
        //~^ ERROR expected tuple struct or tuple variant, found constant `StructConst1`
        _ => { },
    }

    struct SomeStruct(u8);

    const StructConst1 : SomeStruct = SomeStruct(1);
    const StructConst2 : SomeStruct = SomeStruct(2);
}
