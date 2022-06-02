// ignore-compare-mode-nll
// revisions: base nll
// [nll]compile-flags: -Zborrowck=mir

// edition:2018

struct Xyz {
    a: u64,
}

trait Foo {}

impl Xyz {
    async fn do_sth<'a>(
        &'a self, foo: &dyn Foo
    ) -> &dyn Foo
    {
        //[nll]~^ ERROR explicit lifetime required in the type of `foo` [E0621]
        foo
        //[base]~^ ERROR explicit lifetime required in the type of `foo` [E0621]
    }
}

fn main() {}
