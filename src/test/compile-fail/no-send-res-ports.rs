fn main() {
    struct foo {
      let _x: comm::Port<()>;
      new(x: comm::Port<()>) { self._x = x; }
      drop {}
    }
   
    let x = ~mut Some(foo(comm::Port()));

    do task::spawn |move x| { //~ ERROR not a sendable value
        let mut y = None;
        *x <-> y;
        log(error, y);
    }
}
