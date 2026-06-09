//@ build-fail
//@ compile-flags: --diagnostic-width=100 -Zwrite-long-types-to-disk=yes

trait Mirror {
    type Image;
}

impl<T> Mirror for T { type Image = T; }

trait Foo {
    fn recurse(&self);
}

impl<T> Foo for T {
    #[allow(unconditional_recursion)]
    fn recurse(&self) {
        (self, self).recurse();
        //~^ ERROR reached the recursion limit while instantiating
    }
}

fn main() {
    ().recurse();
}
