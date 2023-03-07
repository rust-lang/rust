pub use reexport::Reexported;

pub struct Foo;
pub enum Bar { X }

pub mod foo {
    pub trait PubPub {
        fn method(&self) {}

        fn method3(&self) {}
    }

    impl PubPub for u32 {}
    impl PubPub for i32 {}
}
pub mod bar {
    trait PubPriv {
        fn method(&self);
    }
}
mod qux {
    pub trait PrivPub {
        fn method(&self);
    }
}
mod quz {
    trait PrivPriv {
        fn method(&self);
    }
}

mod reexport {
    pub trait Reexported {
        fn method(&self);
    }
}
