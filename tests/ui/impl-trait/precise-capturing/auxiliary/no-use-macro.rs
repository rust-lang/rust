// A macro_rules macro in 2015 that has an RPIT without `use<>` that would
// cause a problem with 2024 capturing rules.

#[macro_export]
macro_rules! macro_rpit {
    () => {
        fn test_mbe(x: &Vec<i32>) -> impl std::fmt::Display {
            x[0]
        }

        pub fn from_mbe() {
            let mut x = vec![];
            x.push(1);

            let element = test_mbe(&x);
            x.push(2);
            println!("{element}");
        }
    };
}
