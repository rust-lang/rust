fn main() {
    match [~"foo", ~"bar", ~"baz"] {
        [a, _, _, .._] => { io::println(a); }
        [~"foo", ~"bar"] => { } //~ ERROR unreachable pattern
        _ => { }
    }

    match ['a', 'b', 'c'] {
        ['a', 'b', 'c', .._tail] => {}
        ['a', 'b', 'c'] => {} //~ ERROR unreachable pattern
        _ => {}
    }
}
