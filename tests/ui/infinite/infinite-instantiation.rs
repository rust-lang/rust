//@ build-fail
//@ compile-flags: --diagnostic-width=100 -Zwrite-long-types-to-disk=yes

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
