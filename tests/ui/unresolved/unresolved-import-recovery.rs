// Check that unresolved imports do not create additional errors and ICEs

mod m {
    pub use crate::unresolved; //~ ERROR unresolved import `crate::unresolved`

    fn f() {
        let unresolved = 0; // OK
    }
}

fn main() {
    match 0u8 {
        m::unresolved => {} // OK
        m::unresolved(..) => {} // OK
        m::unresolved{..} => {} // OK
    }
}
