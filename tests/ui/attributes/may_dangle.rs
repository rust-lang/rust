#![feature(dropck_eyepatch)]

struct Implee1<'a, T, const N: usize>(&'a T);
struct Implee2<'a, T, const N: usize>(&'a T);
struct Implee3<'a, T, const N: usize>(&'a T);
trait NotDrop {}

unsafe impl<#[may_dangle] 'a, T, const N: usize> NotDrop for Implee1<'a, T, N> {}
//~^ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl

unsafe impl<'a, #[may_dangle] T, const N: usize> NotDrop for Implee2<'a, T, N> {}
//~^ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl

unsafe impl<'a, T, #[may_dangle] const N: usize> Drop for Implee1<'a, T, N> {
    //~^ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl
    fn drop(&mut self) {}
}

// Ok, lifetime param in a `Drop` impl.
unsafe impl<#[may_dangle] 'a, T, const N: usize> Drop for Implee2<'a, T, N> {
    fn drop(&mut self) {}
}

// Ok, type param in a `Drop` impl.
unsafe impl<'a, #[may_dangle] T, const N: usize> Drop for Implee3<'a, T, N> {
    fn drop(&mut self) {}
}

// Check that this check is not textual.
mod fake {
    trait Drop {
        fn drop(&mut self);
    }
    struct Implee<T>(T);

    unsafe impl<#[may_dangle] T> Drop for Implee<T> {
        //~^ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl
        fn drop(&mut self) {}
    }
}

#[may_dangle] //~ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl
struct Dangling;

#[may_dangle] //~ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl
impl NotDrop for () {
}

#[may_dangle] //~ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl
fn main() {
    #[may_dangle] //~ ERROR must be applied to a lifetime or type generic parameter in `Drop` impl
    let () = ();
}
