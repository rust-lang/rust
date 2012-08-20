// error-pattern:explicit failure

fn foo(s: ~str) { }

fn main() {
    let i =
        match Some::<int>(3) { None::<int> => { fail } Some::<int>(_) => { fail } };
    foo(i);
}
