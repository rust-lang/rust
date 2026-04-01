//@ check-pass

fn main() {}

struct RawTableInner<A> {
    alloc: A,
}

impl<A> RawTableInner<A> {
    fn prepare_resize(
        self,
    ) -> ScopeGuard<Self, impl FnMut(&mut Self)> {
        ScopeGuard { dropfn: move |self_| {}, value: self,  }
    }
}

pub struct ScopeGuard<T, F>
where
    F: FnMut(&mut T),
{
    dropfn: F,
    value: T,
}
