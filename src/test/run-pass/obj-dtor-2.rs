

obj foo(x: @mutable int) {drop { log "running dtor"; *x = *x + 1; } }

fn main() {
    let mbox = @mutable 10;
    { let x = foo(mbox); }
    assert (*mbox == 11);
}