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

// The cycle ends up with the same args. Because of the `'x` bound, we don't
// need to treat `'y` as live.
fn same_args<T: Mirrored>(x: &String) {
    let s = String::from("hi");
    let a = make_mirrored::<T>(x, &s);
    drop(s);
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

// The cycle ends up with different args. Given that neither are `'x`, we don't
// technically need to treat `'y` and `'z` as live. However, this is complicated
// to reason about.
fn differing_args<T: Rotating>(x: &String, o: &String) {
    let s = String::from("hi");
    let a = make_rotating::<T>(x, &s, o);
    drop(s);
    //~^ ERROR cannot move out of `s`
    drop(a);
}

// Repeated instances of the same alias definition outside a cycle must be
// visited separately when their arguments differ.
mod repeated_outside_cycle {
    trait Tr {
        type A<'x>;
    }

    fn mk<'x, T: Tr>(_: &'x String) -> T::A<'x> {
        loop {}
    }

    pub fn borrow_second<T: Tr>(o: &String) {
        let s = String::from("hi");
        let pair = (mk::<T>(o), mk::<T>(&s));
        drop(s);
        //~^ ERROR cannot move out of `s` because it is borrowed
        drop(pair);
    }

    pub fn borrow_first<T: Tr>(o: &String) {
        let s = String::from("hi");
        let pair = (mk::<T>(&s), mk::<T>(o));
        drop(s);
        //~^ ERROR cannot move out of `s` because it is borrowed
        drop(pair);
    }
}

fn main() {}
