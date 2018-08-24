#![crate_name = "foo"]

use std::io::Read;
use std::borrow::Borrow;

// @has foo/fn.foo.html
// @has - //pre 'foo('
// @matches - '_x: impl <a class="trait" href="[^"]+/trait\.Clone\.html"'
// @matches - '_z: .+impl.+trait\.Copy\.html.+, impl.+trait\.Clone\.html'
pub fn foo(_x: impl Clone, _y: i32, _z: (impl Copy, impl Clone)) {
}

pub trait Trait {
    // @has foo/trait.Trait.html
    // @has - 'method</a>('
    // @matches - '_x: impl <a class="trait" href="[^"]+/trait\.Debug\.html"'
    fn method(&self, _x: impl std::fmt::Debug) {
    }
}

pub struct S<T>(T);

impl<T> S<T> {
    // @has foo/struct.S.html
    // @has - 'bar</a>('
    // @matches - '_bar: impl <a class="trait" href="[^"]+/trait\.Copy\.html"'
    pub fn bar(_bar: impl Copy) {
    }

    // @has - 'baz</a>('
    // @matches - '_baz:.+struct\.S\.html.+impl .+trait\.Clone\.html'
    pub fn baz(_baz: S<impl Clone>) {
    }

    // @has - 'qux</a>('
    // @matches - 'trait\.Read\.html'
    pub fn qux(_qux: impl IntoIterator<Item = S<impl Read>>) {
    }
}

// @has - 'method</a>('
// @matches - '_x: impl <a class="trait" href="[^"]+/trait\.Debug\.html"'
impl<T> Trait for S<T> {}

// @has foo/fn.much_universe.html
// @matches - 'T:.+Borrow.+impl .+trait\.Trait\.html'
// @matches - 'U:.+IntoIterator.+= impl.+Iterator\.html.+= impl.+Clone\.html'
// @matches - '_: impl .+trait\.Read\.html.+ \+ .+trait\.Clone\.html'
pub fn much_universe<
    T: Borrow<impl Trait>,
    U: IntoIterator<Item = impl Iterator<Item = impl Clone>>,
>(
    _: impl Read + Clone,
) {
}
