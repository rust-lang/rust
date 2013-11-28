fn call_f(f: proc() -> int) -> int {
    f()
}

fn main() {
    let t = ~3;

    call_f(proc() { *t + 1 });
    call_f(proc() { *t + 1 }); //~ ERROR capture of moved value
}
