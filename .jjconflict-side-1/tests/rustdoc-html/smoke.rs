//@ has smoke/index.html

//! Very docs

//@ has smoke/bar/index.html
pub mod bar {

    /// So correct
    //@ has smoke/bar/baz/index.html
    pub mod baz {
        /// Much detail
        //@ has smoke/bar/baz/fn.baz.html
        pub fn baz() { }
    }

    /// *wow*
    //@ has smoke/bar/trait.Doge.html
    pub trait Doge { fn dummy(&self) { } }

    //@ has smoke/bar/struct.Foo.html
    pub struct Foo { x: isize, y: usize }

    //@ has smoke/bar/fn.prawns.html
    pub fn prawns((a, b): (isize, usize), Foo { x, y }: Foo) { }
}
