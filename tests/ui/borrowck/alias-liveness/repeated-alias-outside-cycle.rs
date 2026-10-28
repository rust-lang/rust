// When calculating liveness for aliases, we need to handle cycles carefully. If
// we see two of the same alias outside of a cycle though, we still need to
// consider liveness within separately.

trait Tr {
    type A<'x>;
}

fn mk<'x, T: Tr>(_: &'x String) -> T::A<'x> {
    loop {}
}

fn borrow_second<T: Tr>(o: &String) {
    let s = String::from("hi");
    let pair = (mk::<T>(o), mk::<T>(&s));
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(pair);
}

fn borrow_first<T: Tr>(o: &String) {
    let s = String::from("hi");
    let pair = (mk::<T>(&s), mk::<T>(o));
    drop(s);
    //~^ ERROR cannot move out of `s` because it is borrowed
    drop(pair);
}

fn main() {}
