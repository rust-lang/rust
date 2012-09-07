type recbox<T: Copy> = {x: ~T};

fn reclift<T: Copy>(t: T) -> recbox<T> { return {x: ~t}; }

fn main() {
    let foo: int = 17;
    let rbfoo: recbox<int> = reclift::<int>(foo);
    assert (*rbfoo.x == foo);
}
