//prior to fixing `everybody_loops` to preserve items, rustdoc would crash on this file, as it
//didn't see that `SomeStruct` implemented `Clone`

//FIXME(misdreavus): whenever rustdoc shows traits impl'd inside bodies, make sure this test
//reflects that

pub struct Bounded<T: Clone>(T);

pub struct SomeStruct;

fn asdf() -> Bounded<SomeStruct> {
    impl Clone for SomeStruct {
        fn clone(&self) -> SomeStruct {
            SomeStruct
        }
    }

    Bounded(SomeStruct)
}
