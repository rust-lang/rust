// error-pattern:may alias with argument

fn foo(x: {mut x: int}, f: fn@()) { log(debug, x); }

fn whoknows(x: @mut {mut x: int}) { *x = {mut x: 10}; }

fn main() {
    let box = @mut {mut x: 1};
    foo(*box, bind whoknows(box));
}
