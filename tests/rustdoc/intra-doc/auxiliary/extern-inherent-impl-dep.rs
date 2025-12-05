#[derive(Clone)]
pub struct PublicStruct;

mod inner {
    use super::PublicStruct;

    impl PublicStruct {
        /// [PublicStruct::clone]
        pub fn method() {}
    }
}
