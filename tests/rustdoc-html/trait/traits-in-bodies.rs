//prior to fixing `everybody_loops` to preserve items, rustdoc would crash on this file, as it
//didn't see that `SomeStruct` implemented `Clone`

pub struct Bounded<T: Clone>(T);

//@ has traits_in_bodies/struct.SomeStruct.html
//@ has - '//h3[@class="code-header"]' 'impl Clone for SomeStruct'
pub struct SomeStruct;

fn asdf() -> Bounded<SomeStruct> {
    impl Clone for SomeStruct {
        fn clone(&self) -> SomeStruct {
            SomeStruct
        }
    }

    Bounded(SomeStruct)
}

//@ has traits_in_bodies/struct.Point.html
//@ has - '//h3[@class="code-header"]' 'impl Copy for Point'
#[derive(Clone)]
pub struct Point {
    x: i32,
    y: i32,
}

const _FOO: () = {
    impl Copy for Point {}
    ()
};

//@ has traits_in_bodies/struct.Inception.html
//@ has - '//h3[@class="code-header"]' 'impl Clone for Inception'
pub struct Inception;

static _BAR: usize = {
    trait HiddenTrait {
        fn hidden_fn(&self) {
            for _ in 0..5 {
                impl Clone for Inception {
                    fn clone(&self) -> Self {
                        // we need to go deeper
                        Inception
                    }
                }
            }
        }
    }

    5
};
