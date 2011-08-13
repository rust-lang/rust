

type recbox<T> = {x: @T};

fn reclift<T>(t: &T) -> recbox<T> { ret {x: @t}; }

fn main() {
    let foo: int = 17;
    let rbfoo: recbox<int> = reclift::<int>(foo);
    assert (*rbfoo.x == foo);
}
