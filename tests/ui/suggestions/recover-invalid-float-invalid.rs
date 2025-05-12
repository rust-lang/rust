// Check that suggestions to add a zero to integers with a preceding dot only appear when the change
// will result in a valid floating point literal.

fn main() {}

fn a() {
    _ = .3u32;
    //~^ ERROR expected expression, found `.`
}

fn b() {
    _ = .0b0;
    //~^ ERROR expected expression, found `.`
}

fn c() {
    _ = .0o07;
    //~^ ERROR expected expression, found `.`
}

fn d() {
    _ = .0x0ABC;
    //~^ ERROR expected expression, found `.`
}
