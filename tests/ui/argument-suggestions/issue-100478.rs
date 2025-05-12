use std::sync::Arc;
macro_rules! GenT {
    ($name:tt) => {
        #[derive(Default, Debug)]
        struct $name {
            #[allow(unused)]
            val: i32,
        }

        impl $name {
            #[allow(unused)]
            fn new(val: i32) -> Self {
                $name { val }
            }
        }
    };
}

GenT!(T1);
GenT!(T2);
GenT!(T3);
GenT!(T4);
GenT!(T5);
GenT!(T6);
GenT!(T7);
GenT!(T8);

#[allow(unused)]
fn foo(p1: T1, p2: Arc<T2>, p3: T3, p4: Arc<T4>, p5: T5, p6: T6, p7: T7, p8: Arc<T8>) {}
fn three_diff(_a: T1, _b: T2, _c: T3) {}
fn four_shuffle(_a: T1, _b: T2, _c: T3, _d: T4) {}

fn main() {
    three_diff(T2::new(0)); //~ ERROR function takes
    four_shuffle(T3::default(), T4::default(), T1::default(), T2::default()); //~ ERROR 35:5: 35:17: arguments to this function are incorrect [E0308]
    four_shuffle(T3::default(), T2::default(), T1::default(), T3::default()); //~ ERROR 36:5: 36:17: arguments to this function are incorrect [E0308]

    let p1 = T1::new(0);
    let p2 = Arc::new(T2::new(0));
    let p3 = T3::new(0);
    let p4 = Arc::new(T4::new(1));
    let p5 = T5::new(0);
    let p6 = T6::new(0);
    let p7 = T7::new(0);
    let p8 = Arc::default();

    foo(
        //~^ ERROR this function takes 8 arguments but 7 arguments were supplied [E0061]
        p1, //p2,
        p3, p4, p5, p6, p7, p8,
    );
}
