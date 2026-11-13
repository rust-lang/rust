// We can get a cycle when calculating liveness for aliases. We obviously need
// to avoid infinite recursion, but we also need to be careful to ensure that we
// mark necessary regions as live, for soundness.

trait Mirror {
    type Other;
}

trait Sym<'a> {
    type C: Sym<'a> + PartialEq<<Self::C as Sym<'a>>::C>;
}

fn make_sym<'a, T: Sym<'a>>(_: &'a String) -> T::C {
    unimplemented!()
}

// An infinitely growing cycle, but we still need to mark `'a` as live.
fn grows_in_self_type<T: for<'a> Sym<'a>>() {
    let s = String::from("hi");
    let keep = make_sym::<'_, T>(&s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(keep);
}

trait Mirrored {
    type A<'x, 'y>: 'x + Mirror<Other = Self::A<'x, 'y>>;
}

fn make_mirrored<'x, 'y, T: Mirrored>(_: &'x String, _: &'y String) -> T::A<'x, 'y> {
    loop {}
}

// The cycle ends up with the same args. We *could* be more precise and treat
// `'y` as not-live, but we're conservative and don't.
fn same_args<T: Mirrored>(x: &String) {
    let s = String::from("hi");
    let a = make_mirrored::<T>(x, &s);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(a);
}

trait Rotating {
    type A<'x, 'y, 'z, Z>: 'x + Mirror<Other = Self::A<'x, 'z, 'y, Vec<Z>>>;
}

fn make_rotating<'x, 'y, 'z, T: Rotating>(
    _: &'x String,
    _: &'y String,
    _: &'z String,
) -> T::A<'x, 'y, 'z, u8> {
    loop {}
}

// The cycle ends up with different args, so we must treat both `'y` and `'z` as live.
fn differing_args<T: Rotating>(x: &String, o: &String) {
    let s = String::from("hi");
    let a = make_rotating::<T>(x, &s, o);
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(a);
}

fn main() {}
