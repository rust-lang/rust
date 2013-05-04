#[crate_type = "lib"];

pub struct Fish {
    x: int
}

mod unexported {
    use super::Fish;
    impl Eq for Fish {
        fn eq(&self, _: &Fish) -> bool { true }
        fn ne(&self, _: &Fish) -> bool { false }
    }
}
