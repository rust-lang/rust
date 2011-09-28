// error-pattern: cannot copy pinned type 'a

resource r(i: @mutable int) {
    *i = *i + 1;
}

fn movearg<T>(i: T) {
    // Implicit copy to mutate reference i
    let j <- i;
}

fn main() {
    let i = @mutable 0;
    {
        let j <- r(i);
        movearg(j);
    }
    log_err *i;
    // nooooooo. destructor ran twice
    assert *i == 2;
}