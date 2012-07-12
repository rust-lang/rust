// Regression test for issue #2783

fn foo(f: fn()) { f() }

fn main() {
    "" || 42; //~ ERROR binary operation || cannot be applied to type `str/~`
    foo || {}; //~ ERROR binary operation || cannot be applied to type `extern fn(fn())`
    //~^ NOTE did you forget the 'do' keyword for the call?
}
