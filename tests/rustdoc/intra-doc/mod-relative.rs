pub mod wrapper {

    pub struct Test<'a> {
        data: &'a (),
    }

    impl<'a> Test<'a> {
        pub fn do_test(&self) {}
    }

    //@ has mod_relative/wrapper/demo/index.html
    //@ has - '//a/@href' '../struct.Test.html#method.do_test'
    /// [`Test::do_test`]
    pub mod demo {
    }

}
