// error-pattern:may alias with argument

fn foo(x: {mutable x: int}, f: fn()) { log x; }

fn whoknows(x: @mutable {mutable x: int}) { *x = {mutable x: 10}; }

fn main() {
    let box = @mutable {mutable x: 1};
    foo(*box, bind whoknows(box));
}
