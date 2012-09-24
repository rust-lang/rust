// xfail-test iloops with optimizations on

// NB: Not sure why this works. I expect the box argument to leak when
// we run out of stack. Maybe the box annihilator works it out?

// error-pattern:ran out of stack
fn main() {
    eat(move @0);
}

fn eat(
    +a: @int
) {
    eat(move a)
}
