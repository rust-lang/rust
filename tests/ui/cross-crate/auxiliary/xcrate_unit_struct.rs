#![crate_type = "lib"]

// used by the rpass test

#[derive(Copy, Clone)]
pub struct Struct;

#[derive(Copy, Clone)]
pub enum Unit {
    UnitVariant,
    Argument(Struct)
}

#[derive(Copy, Clone)]
pub struct TupleStruct(pub usize, pub &'static str);

// used by the cfail test

#[derive(Copy, Clone)]
pub struct StructWithFields {
    pub foo: isize,
}

#[derive(Copy, Clone)]
pub struct StructWithPrivFields {
    foo: isize,
}

#[derive(Copy, Clone)]
pub enum EnumWithVariants {
    EnumVariant,
    EnumVariantArg(isize)
}
