fn main() {
    class foo {
      let _x: comm::port<()>;
      new(x: comm::port<()>) { self._x = x; }
      drop {}
    }
   
    let x = ~mut some(foo(comm::port()));

    do task::spawn |move x| { //! ERROR not a sendable value
        let mut y = none;
        *x <-> y;
        log(error, y);
    }
}
