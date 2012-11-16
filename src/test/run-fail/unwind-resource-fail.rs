// error-pattern:squirrel

struct r {
    i: int,
    drop { fail ~"squirrel" }
}

fn r(i: int) -> r { r { i: i } }

fn main() {
    @0;
    let r = move r(0);
}
