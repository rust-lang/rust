struct dtor {
    x: @mut int,

    drop {
        // abuse access to shared mutable state to write this code
        *self.x -= 1;
    }
}

fn unwrap<T>(+o: Option<T>) -> T {
    match move o {
      Some(move v) => v,
      None => fail
    }
}

fn main() {
    let x = @mut 1;

    {
        let b = Some(dtor { x:x });
        let c = unwrap(b);
    }

    assert *x == 0;
}