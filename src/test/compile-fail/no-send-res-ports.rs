fn main() {
    struct foo {
      let _x: comm::Port<()>;
      new(x: comm::Port<()>) { self._x = x; }
      drop {}
    }
   
    let x = ~mut some(foo(comm::port()));

    do task::spawn |move x| { //~ ERROR not a sendable value
        let mut y = none;
        *x <-> y;
        log(error, y);
    }
}
