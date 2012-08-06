// error-pattern:explicit failure

fn foo(s: ~str) { }

fn main() {
    let i =
        match some::<int>(3) { none::<int> => { fail } some::<int>(_) => { fail } };
    foo(i);
}
