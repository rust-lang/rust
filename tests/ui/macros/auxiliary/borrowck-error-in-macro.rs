#[macro_export]
macro_rules! ice {
    () => {
        fn main() {
            let d = &mut 0;
            let c = || *d += 1;
            c();
        }
    };
}
