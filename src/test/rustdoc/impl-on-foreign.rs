#![crate_name = "foo"]

use std::convert::AsRef;
pub struct Gus;

// @has 'foo/struct.Gus.html'
// @has - '//h3[@class="impl"]/code' 'impl AsRef<str> for Gus'
impl AsRef<str> for Gus {
    fn as_ref(&self) -> &str {
        todo!()
    }
}

// @has - '//h3[@class="impl"]/code' 'impl AsRef<Gus> for str'
impl AsRef<Gus> for str {
    fn as_ref(&self) -> &Gus {
        todo!()
    }
}

// @has - '//h3[@class="impl"]/code' 'impl AsRef<Gus> for String'
impl AsRef<Gus> for String {
    fn as_ref(&self) -> &Gus {
        todo!()
    }
}
