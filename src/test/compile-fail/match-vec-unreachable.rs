fn main() {
    let x: ~[(int, int)] = ~[];
    match x {
        [a, (2, 3), _] => (),
        [(1, 2), (2, 3), b] => (), //~ ERROR unreachable pattern
        _ => ()
    }

    match [~"foo", ~"bar", ~"baz"] {
        [a, _, _, .._] => { println(a); }
        [~"foo", ~"bar", ~"baz", ~"foo", ~"bar"] => { } //~ ERROR unreachable pattern
        _ => { }
    }

    match ['a', 'b', 'c'] {
        ['a', 'b', 'c', .._tail] => {}
        ['a', 'b', 'c'] => {} //~ ERROR unreachable pattern
        _ => {}
    }
}
