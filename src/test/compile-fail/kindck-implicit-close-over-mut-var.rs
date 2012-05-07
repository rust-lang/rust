fn use(_i: int) {}

fn foo() {
    // Here, i is *moved* into the closure: OK
    let mut i = 0;
    task::spawn {||
        use(i);
    }
}

fn bar() {
    // Here, i would be implicitly *copied* but it
    // is mutable: bad
    let mut i = 0;
    while i < 10 {
        task::spawn {||
            use(i); //! ERROR mutable variables cannot be implicitly captured
        }
        i += 1;
    }
}

fn car() {
    // Here, i is mutable, but *explicitly* copied:
    let mut i = 0;
    while i < 10 {
        task::spawn {|copy i|
            use(i);
        }
        i += 1;
    }
}

fn main() {
}