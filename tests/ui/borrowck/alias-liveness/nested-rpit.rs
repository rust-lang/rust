//@ edition: 2024
//@ compile-flags: --crate-type lib
//@ check-pass

struct ScopeGuard<T, F>
where
    F: FnMut(&mut T),
{
    dropfn: F,
    value: T,
}

impl<T, F> core::ops::Deref for ScopeGuard<T, F>
where
    F: FnMut(&mut T),
{
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T, F> core::ops::DerefMut for ScopeGuard<T, F>
where
    F: FnMut(&mut T),
{
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<T, F> Drop for ScopeGuard<T, F>
where
    F: FnMut(&mut T),
{
    #[inline]
    fn drop(&mut self) {
        (self.dropfn)(&mut self.value);
    }
}

struct RawTableInner;

impl RawTableInner {
    fn prepare_resize<'a, A>(
        &self,
        alloc: &'a A,
    ) -> Result<ScopeGuard<Self, impl FnMut(&mut Self) + 'a>, ()>
    {
        Ok(ScopeGuard {
            value: RawTableInner,
            dropfn: move |self_| {},
        })
    }

    fn resize_inner<A>(
        &mut self,
        alloc: &A,
        hasher: &dyn Fn(&mut Self, usize) -> u64,
    ) -> Result<(), ()>
    {
        let mut new_table = self.prepare_resize(alloc)?;

        for _ in 0..10 {
            hasher(self, 0);
            new_table.prepare_insert_index();
        }

        core::mem::swap(self, &mut new_table);

        Ok(())
    }

    fn prepare_insert_index(&mut self) {}
}
