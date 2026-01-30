struct Priv;

pub trait Super {
    type AssocSuper: GetUnreachable;
}
#[expect(private_bounds)]
pub trait Sub: Super<AssocSuper = Priv> {}

// This Dummy type is only used in call_handler
struct Dummy;
impl Super for Dummy {
    type AssocSuper = Priv;
}
impl Sub for Dummy {}

pub trait SubHandler {
    fn handle<T: Sub>();
}
pub fn call_handler<T: SubHandler>() {
    <T as SubHandler>::handle::<Dummy>();
}

pub trait GetUnreachable {
    type Assoc;
}
mod m {
    pub struct Unreachable;
    impl Unreachable {
        #[expect(dead_code)]
        pub fn generic<T>() {}
    }
    impl crate::GetUnreachable for crate::Priv {
        type Assoc = Unreachable;
    }
}
