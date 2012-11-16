// xfail-test leaks
// error-pattern:wombat

struct r {
    i: int,
    drop { fail ~"wombat" }
}

fn r(i: int) -> r { r { i: i } }

fn main() {
    @0;
    let r = move r(0);
    fail;
}