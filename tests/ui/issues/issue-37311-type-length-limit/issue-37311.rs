//@ build-fail
// The regex below normalizes the long type file name to make it suitable for compare-modes.
//@ normalize-stderr: "'\$TEST_BUILD_DIR/.*\.long-type.txt'" -> "'$$TEST_BUILD_DIR/$$FILE.long-type.txt'"

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
