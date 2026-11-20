// The goal of this test is to ensure that the attributes of all imports are taken into
// account.

#![crate_name = "foo"]

mod a {
    /// 1
    pub struct Type;
}

mod b {
    /// 2
    pub use crate::a::Type;
}

mod c {
    /// 3
    pub use crate::b::Type;
    /// 4
    pub use crate::b::Type as Woof;
}

//@ has 'foo/struct.Type.html'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'foo 2 1'
/// foo
pub use b::Type;
//@ has 'foo/struct.Whatever.html'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'whatever 3 2 1'
/// whatever
pub use c::Type as Whatever;
//@ has 'foo/struct.Woof.html'
//@ has - '//*[@class="toggle top-doc"]/*[@class="docblock"]' 'a dog 4 2 1'
/// a dog
pub use c::Woof;
