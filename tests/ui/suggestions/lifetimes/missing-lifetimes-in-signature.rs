pub trait Get<T> {
    fn get(self) -> T;
}

struct Foo {
    x: usize,
}

impl Get<usize> for Foo {
    fn get(self) -> usize {
        self.x
    }
}

fn foo<G, T>(g: G, dest: &mut T) -> impl FnOnce()
where
    G: Get<T>,
{
    move || {
        //~^ ERROR hidden type for `impl FnOnce()` captures lifetime
        *dest = g.get();
    }
}

// After applying suggestion for `foo`:
fn bar<G, T>(g: G, dest: &mut T) -> impl FnOnce() + '_
where
    G: Get<T>,
{
    move || {
        //~^ ERROR the parameter type `G` may not live long enough
        *dest = g.get();
    }
}

// After applying suggestion for `bar`:
fn baz<G: 'a, T>(g: G, dest: &mut T) -> impl FnOnce() + '_
//~^ ERROR undeclared lifetime name `'a`
where
    G: Get<T>,
{
    move || {
        *dest = g.get();
    }
}

// After applying suggestion for `baz`:
fn qux<'a, G: 'a, T>(g: G, dest: &mut T) -> impl FnOnce() + '_
where
    G: Get<T>,
{
    move || {
        //~^ ERROR the parameter type `G` may not live long enough
        *dest = g.get();
    }
}

// Same as above, but show that we pay attention to lifetime names from parent item
impl<'a> Foo {
    fn qux<'b, G: Get<T> + 'b, T>(g: G, dest: &mut T) -> impl FnOnce() + '_ {
        move || {
            //~^ ERROR the parameter type `G` may not live long enough
            *dest = g.get();
        }
    }
}

// After applying suggestion for `qux`:
fn bat<'a, G: 'a, T>(g: G, dest: &mut T) -> impl FnOnce() + '_ + 'a
where
    G: Get<T>,
{
    move || {
        //~^ ERROR the parameter type `G` may not live long enough
        //~| ERROR explicit lifetime required
        *dest = g.get();
    }
}

// Potential incorrect attempt:
fn bak<'a, G, T>(g: G, dest: &'a mut T) -> impl FnOnce() + 'a
where
    G: Get<T>,
{
    move || {
        //~^ ERROR the parameter type `G` may not live long enough
        *dest = g.get();
    }
}

// We need to tie the lifetime of `G` with the lifetime of `&mut T` and the returned closure:
fn ok<'a, G: 'a, T>(g: G, dest: &'a mut T) -> impl FnOnce() + 'a
where
    G: Get<T>,
{
    move || {
        *dest = g.get();
    }
}

// This also works. The `'_` isn't necessary but it's where we arrive to following the suggestions:
fn ok2<'a, G: 'a, T>(g: G, dest: &'a mut T) -> impl FnOnce() + '_ + 'a
where
    G: Get<T>,
{
    move || {
        *dest = g.get();
    }
}

fn main() {}
