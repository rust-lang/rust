//@ run-pass

fn foo<const N: usize>(_: [u8; N]) -> [u8; N] {
    [0; N]
}

fn bar() {
    let _x: [u8; 3] = [0; _];
    let _y: [u8; _] = [0; 3];
}

fn main() {
    let _x = foo::<_>([1, 2]);
    bar();
}
