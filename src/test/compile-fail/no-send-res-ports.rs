fn main() {
    struct foo {
      _x: comm::Port<()>,
    }

    impl foo : Drop {
        fn finalize() {}
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
