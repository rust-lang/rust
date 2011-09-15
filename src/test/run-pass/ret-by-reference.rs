tag option<T> { some(T); none; }

fn get<@T>(opt: option<T>) -> &T {
    alt opt {
      some(x) { ret x; }
    }
}

fn get_mut(a: {mutable x: @int}, _b: int) -> &!0 @int {
    ret a.x;
}

fn main() {
    let x = some(@50);
    let &y = get(x);
    assert *y == 50;
    assert get(some(10)) == 10;

    let y = {mutable x: @50};
    let &box = get_mut(y, 4);
    assert *box == 50;
    assert *get_mut({mutable x: @70}, 5) == 70;
}
