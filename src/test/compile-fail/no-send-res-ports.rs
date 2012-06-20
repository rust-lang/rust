fn main() {
    resource foo(_x: comm::port<()>) {}

    let x = ~mut some(foo(comm::port()));

    task::spawn {|move x| //! ERROR not a sendable value
        let mut y = none;
        *x <-> y;
        log(error, y);
    }
}
