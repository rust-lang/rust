tag option<T> { some(T); none; }

fn get<T>(opt: option<T>) -> &T {
    alt opt {
      some(x) { ret x; }
    }
}

fn get_mut(a: {mutable x: @int}, _b: int) -> &!1 @int {
    ret a.x;
}

fn get_deep(a: {mutable y: {mutable x: @int}}) -> &!@int {
    ret get_mut(a.y, 1);
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

    let u = {mutable y: {mutable x: @10}};
    let &deep = get_deep(u);
    assert *deep == 10;
    assert *get_deep({mutable y: {mutable x: @11}}) + 2 == 13;
}
