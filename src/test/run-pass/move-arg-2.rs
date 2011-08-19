fn test(foo: -@[int]) { assert (foo[0] == 10); }

fn main() {
    let x = @[10];
    // Test forgetting a local by move-in
    test(x);

    // Test forgetting a temporary by move-in.
    test(@[10]);
}
