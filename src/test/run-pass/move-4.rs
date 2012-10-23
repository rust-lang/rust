
extern mod std;

fn test(foo: @{a: int, b: int, c: int}) -> @{a: int, b: int, c: int} {
    let foo = foo;
    let bar = move foo;
    let baz = move bar;
    let quux = move baz;
    return quux;
}

fn main() { let x = @{a: 1, b: 2, c: 3}; let y = test(x); assert (y.c == 3); }
