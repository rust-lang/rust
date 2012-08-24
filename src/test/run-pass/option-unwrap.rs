struct dtor {
    x: @mut int;

    drop {
        // abuse access to shared mutable state to write this code
        *self.x -= 1;
    }
}

fn unwrap<T>(+o: option<T>) -> T {
    match move o {
      some(move v) => v,
      none => fail
    }
}

fn main() {
    let x = @mut 1;

    {
        let b = some(dtor { x:x });
        let c = unwrap(b);
    }

    assert *x == 0;
}