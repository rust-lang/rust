fn main() {
    let x: ~[(int, int)] = ~[];
    match x {
        [(1, 2), (2, 3), _] => (),
        [a, _, (3, 4)] => (), //~ ERROR unreachable pattern
        _ => ()
    }

    match [~"foo", ~"bar", ~"baz"] {
        [a, _, _, .._] => { io::println(a); }
        [~"foo", ~"bar", ~"baz", ~"foo", ~"bar"] => { } //~ ERROR unreachable pattern
        _ => { }
    }

    match ['a', 'b', 'c'] {
        ['a', 'b', 'c', .._tail] => {}
        ['a', 'b', 'c'] => {} //~ ERROR unreachable pattern
        _ => {}
    }
}
