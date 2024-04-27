#![crate_type = "lib"]

pub struct Fish {
    pub x: isize
}

mod unexported {
    use super::Fish;
    impl PartialEq for Fish {
        fn eq(&self, _: &Fish) -> bool { true }
        fn ne(&self, _: &Fish) -> bool { false }
    }
}
