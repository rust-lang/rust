use std;

fn ack(m: int, n: int) -> int {
    if m == 0 {
        ret n + 1
    } else {
        if n == 0 {
            ret ack(m - 1, 1);
        } else {
            ret ack(m - 1, ack(m, n - 1));
        }
    }
}

fn main(args: [str]) {
    // FIXME: #1527
    sys::set_min_stack(1000000u);
    let n = if vec::len(args) == 2u {
        int::from_str(args[1])
    } else {
        11
    };
    std::io::println(#fmt("Ack(3,%d): %d\n", n, ack(3, n)));
}
