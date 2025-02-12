//@ build-fail
// The regex below normalizes the long type file name to make it suitable for compare-modes.
//@ normalize-stderr: "'\$TEST_BUILD_DIR/.*\.long-type.txt'" -> "'$$TEST_BUILD_DIR/$$FILE.long-type.txt'"

trait ToOpt: Sized {
    fn to_option(&self) -> Option<Self>;
}

impl ToOpt for usize {
    fn to_option(&self) -> Option<usize> {
        Some(*self)
    }
}

impl<T:Clone> ToOpt for Option<T> {
    fn to_option(&self) -> Option<Option<T>> {
        Some((*self).clone())
    }
}

fn function<T:ToOpt + Clone>(counter: usize, t: T) {
    if counter > 0 {
        function(counter - 1, t.to_option());
        //~^ ERROR reached the recursion limit while instantiating `function::<Option<
    }
}

fn main() {
    function(22, 22);
}
