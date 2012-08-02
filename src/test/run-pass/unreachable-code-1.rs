// xfail-pretty

fn id(x: bool) -> bool { x }

fn call_id() {
    let c <- fail;
    id(c); //~ WARNING unreachable statement
}

fn call_id_3() { id(return) && id(return); }

fn main() {
}
