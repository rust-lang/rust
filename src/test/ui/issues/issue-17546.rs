// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
use foo::MyEnum::Result;
use foo::NoResult; // Through a re-export

mod foo {
    pub use self::MyEnum::NoResult;

    pub enum MyEnum {
        Result,
        NoResult
    }

    fn new() -> NoResult<MyEnum, String> {
        //~^ ERROR expected type, found variant `NoResult`
        unimplemented!()
    }
}

mod bar {
    use foo::MyEnum::Result;
    use foo;

    fn new() -> Result<foo::MyEnum, String> {
        //~^ ERROR expected type, found variant `Result`
        unimplemented!()
    }
}

fn new() -> Result<foo::MyEnum, String> {
    //~^ ERROR expected type, found variant `Result`
    unimplemented!()
}

fn newer() -> NoResult<foo::MyEnum, String> {
    //~^ ERROR expected type, found variant `NoResult`
    unimplemented!()
}

fn main() {
    let _ = new();
}
