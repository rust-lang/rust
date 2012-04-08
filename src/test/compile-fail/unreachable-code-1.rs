// xfail-pretty

fn id(x: bool) -> bool { x }

fn call_id() {
    let c <- fail;
    id(c); //! WARNING unreachable statement
}

fn call_id_3() { id(ret) && id(ret); }
    //!^ ERROR the type of this value must be known

fn main() {
}
