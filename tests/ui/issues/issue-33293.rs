fn main() {
    match 0 {
        aaa::bbb(_) => ()
        //~^ ERROR cannot find item `aaa`
    };
}
