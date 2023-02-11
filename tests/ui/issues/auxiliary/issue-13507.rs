pub mod testtypes {
    use std::any::TypeId;

    pub fn type_ids() -> Vec<TypeId> {
        vec![
            TypeId::of::<FooBool>(),
            TypeId::of::<FooInt>(),
            TypeId::of::<FooUint>(),
            TypeId::of::<FooFloat>(),
            TypeId::of::<FooStr>(),
            TypeId::of::<FooArray>(),
            TypeId::of::<FooSlice>(),
            TypeId::of::<FooBox>(),
            TypeId::of::<FooPtr>(),
            TypeId::of::<FooRef>(),
            TypeId::of::<FooFnPtr>(),
            TypeId::of::<FooNil>(),
            TypeId::of::<FooTuple>(),
            TypeId::of::<FooTrait>(),
            TypeId::of::<FooStruct>(),
            TypeId::of::<FooEnum>()
        ]
    }

    // Tests Bool
    pub type FooBool = bool;

    // Tests Char
    pub type FooChar = char;

    // Tests Int (does not test all variants of IntTy)
    pub type FooInt = isize;

    // Tests Uint (does not test all variants of UintTy)
    pub type FooUint = usize;

    // Tests Float (does not test all variants of FloatTy)
    pub type FooFloat = f64;

    // Tests Str
    pub type FooStr = str;

    // Tests Array
    pub type FooArray = [u8; 1];

    // Tests Slice
    pub type FooSlice = [u8];

    // Tests Box (of u8)
    pub type FooBox = Box<u8>;

    // Tests RawPtr
    pub type FooPtr = *const u8;

    // Tests Ref
    pub type FooRef = &'static u8;

    // Tests FnPtr
    pub type FooFnPtr = fn(u8) -> bool;

    // Tests Dynamic
    pub trait FooTrait {
        fn foo_method(&self) -> usize;
    }

    // Tests struct
    pub struct FooStruct {
        pub pub_foo_field: usize,
        foo_field: usize
    }

    // Tests enum
    pub enum FooEnum {
        VarA(usize),
        VarB(usize, usize)
    }

    // Tests Tuple
    pub type FooNil = ();
    pub type FooTuple = (u8, i8, bool);

    // Skipping Param

    // Skipping Infer

    // Skipping Error
}
