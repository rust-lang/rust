mod a {
    pub trait Trait {}
}

mod b {
    use Trait; //~ ERROR unresolved import `Trait`
}

mod c {
    impl Trait for () {} //~ ERROR cannot find trait `Trait` in this scope
}

fn main() {}
