struct UnitStruct;
struct WrappedUnitStruct(UnitStruct);
struct StructWrappedUnitStruct {
    inner: UnitStruct,
}

struct WrappedPhantomData(std::marker::PhantomData<()>);

struct UnitTuple();
struct WrappedUnitTuple(UnitTuple);

struct EmptyStruct {}
struct WrappedEmptyStruct(EmptyStruct);


fn main() {
    WrappedUnitStruct(());
    //~^ mismatched types [E0308]
    //~| try directly constructing the struct
    StructWrappedUnitStruct {
        inner: 0,
        //~^ mismatched types [E0308]
        //~| try directly constructing the struct
    };
    WrappedPhantomData(());
    //~^ mismatched types [E0308]
    //~| try directly constructing the struct
    WrappedUnitTuple(());
    //~^ mismatched types [E0308]
    //~| try directly constructing the struct
    WrappedEmptyStruct(());
    //~^ mismatched types [E0308]
    //~| try directly constructing the struct
}
