pub mod a {
    pub struct Abc;
    pub enum Def {}
}

pub mod b {

}

pub use self::a::Abc;
