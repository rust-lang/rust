//@ known-bug: #137467
//@ edition: 2021

fn meow(x: (u32, u32, u32)) {
    let f = || {
        let ((0, a, _) | (_, _, a)) = x;
    };
}
