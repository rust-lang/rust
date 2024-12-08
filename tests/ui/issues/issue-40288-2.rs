fn prove_static<T: 'static + ?Sized>(_: &'static T) {}

fn lifetime_transmute_slice<'a, T: ?Sized>(x: &'a T, y: &T) -> &'a T {
    let mut out = [x];
    {
        let slice: &mut [_] = &mut out;
        slice[0] = y;
    }
    out[0]
    //~^ ERROR explicit lifetime required in the type of `y` [E0621]
}

struct Struct<T, U: ?Sized> {
    head: T,
    _tail: U
}

fn lifetime_transmute_struct<'a, T: ?Sized>(x: &'a T, y: &T) -> &'a T {
    let mut out = Struct { head: x, _tail: [()] };
    {
        let dst: &mut Struct<_, [()]> = &mut out;
        dst.head = y;
    }
    out.head
    //~^ ERROR explicit lifetime required in the type of `y` [E0621]
}

fn main() {
    prove_static(lifetime_transmute_slice("", &String::from("foo")));
    prove_static(lifetime_transmute_struct("", &String::from("bar")));
}
