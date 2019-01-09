// Local module shadows `ep_lib` from extern prelude

mod ep_lib {
    pub struct S;

    impl S {
        pub fn internal(&self) {}
    }
}

fn main() {
    let s = ep_lib::S;
    s.internal(); // OK
}
