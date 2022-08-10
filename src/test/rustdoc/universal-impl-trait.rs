#![crate_name = "foo"]

use std::io::Read;
use std::borrow::Borrow;

// @has foo/fn.foo.html
// @has - //pre 'foo('
// @matchestext - '_x: impl <a class="trait" href="[^"]+/trait\.Clone\.html"'
// @matchestext - '_z: .+impl.+trait\.Copy\.html.+, impl.+trait\.Clone\.html'
pub fn foo(_x: impl Clone, _y: i32, _z: (impl Copy, impl Clone)) {
}

pub trait Trait {
    // @has foo/trait.Trait.html
    // @hastext - 'method</a>('
    // @matchestext - '_x: impl <a class="trait" href="[^"]+/trait\.Debug\.html"'
    fn method(&self, _x: impl std::fmt::Debug) {
    }
}

pub struct S<T>(T);

impl<T> S<T> {
    // @has foo/struct.S.html
    // @hastext - 'bar</a>('
    // @matchestext - '_bar: impl <a class="trait" href="[^"]+/trait\.Copy\.html"'
    pub fn bar(_bar: impl Copy) {
    }

    // @hastext - 'baz</a>('
    // @matchestext - '_baz:.+struct\.S\.html.+impl .+trait\.Clone\.html'
    pub fn baz(_baz: S<impl Clone>) {
    }

    // @hastext - 'qux</a>('
    // @matchestext - 'trait\.Read\.html'
    pub fn qux(_qux: impl IntoIterator<Item = S<impl Read>>) {
    }
}

// @hastext - 'method</a>('
// @matchestext - '_x: impl <a class="trait" href="[^"]+/trait\.Debug\.html"'
impl<T> Trait for S<T> {}

// @has foo/fn.much_universe.html
// @matchestext - 'T:.+Borrow.+impl .+trait\.Trait\.html'
// @matchestext - 'U:.+IntoIterator.+= impl.+Iterator\.html.+= impl.+Clone\.html'
// @matchestext - '_: impl .+trait\.Read\.html.+ \+ .+trait\.Clone\.html'
pub fn much_universe<
    T: Borrow<impl Trait>,
    U: IntoIterator<Item = impl Iterator<Item = impl Clone>>,
>(
    _: impl Read + Clone,
) {
}
