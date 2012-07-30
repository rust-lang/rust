// Testing that calling #fmt (via #debug) doesn't complain about impure borrows

pure fn foo() {
    let a = {
        b: @"hi",
        c: 0,
        d: 1,
        e: 'a',
        f: 0.0,
        g: true
    };
    debug!{"test %?", a.b};
    debug!{"test %u", a.c};
    debug!{"test %i", a.d};
    debug!{"test %c", a.e};
    debug!{"test %f", a.f};
    debug!{"test %b", a.g};
}

fn main() {
}