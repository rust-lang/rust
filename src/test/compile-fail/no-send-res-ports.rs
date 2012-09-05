fn main() {
    struct foo {
      let _x: comm::Port<()>;
      drop {}
    }

    fn foo(x: comm::Port<()>) -> foo {
        foo {
            _x: x
        }
    }
   
    let x = ~mut Some(foo(comm::Port()));

    do task::spawn |move x| { //~ ERROR not a sendable value
        let mut y = None;
        *x <-> y;
        log(error, y);
    }
}
