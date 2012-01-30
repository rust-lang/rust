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
    let n = if vec::len(args) == 2u {
        int::from_str(args[1])
    } else {
        8
    };
    std::io::println(#fmt("Ack(3,%d): %d\n", n, ack(3, n)));
}
