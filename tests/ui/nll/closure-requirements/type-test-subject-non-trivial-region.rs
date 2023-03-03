// chek-fail
// known-bug: #108639

trait Trait {
    type Item<'a>: 'a;
}

fn assert_static<T: 'static>(_: T) {}
fn relate<T>(_: T, _: T) {}

fn test_args<I: Trait>() {
    let closure = |a, b| {
        relate(&a, b);
        assert_static(a);
    };
    closure(None::<I::Item<'_>>, &None::<I::Item<'_>>);
}

fn main() {}
