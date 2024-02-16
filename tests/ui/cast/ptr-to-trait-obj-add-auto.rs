// check-fail

trait Trait<'a> {}

fn add_auto<'a>(x: *mut dyn Trait<'a>) -> *mut (dyn Trait<'a> + Send) {
    x as _ //~ error: the trait bound `dyn Trait<'_>: Unsize<dyn Trait<'_> + Send>` is not satisfied
}

fn main() {}
