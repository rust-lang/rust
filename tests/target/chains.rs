// Test chain formatting.

fn main() {
    // Don't put chains on a single line if it wasn't so in source.
    let a = b.c
             .d
             .1
             .foo(|x| x + 1);

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
                       .ddddddddddddddddddddddddddd();

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
                       .ddddddddddddddddddddddddddd
                       .eeeeeeee();

    // Test case where first chain element isn't a path, but is shorter than
    // the size of a tab.
    x().y(|| {
        match cond() {
            true => (),
            false => (),
        }
    });

    loong_func().quux(move || {
        if true {
            1
        } else {
            2
        }
    });

    some_fuuuuuuuuunction().method_call_a(aaaaa, bbbbb, |c| {
        let x = c;
        x
    });

    some_fuuuuuuuuunction()
        .method_call_a(aaaaa, bbbbb, |c| {
            let x = c;
            x
        })
        .method_call_b(aaaaa, bbbbb, |c| {
            let x = c;
            x
        });

    fffffffffffffffffffffffffffffffffff(a, {
        SCRIPT_TASK_ROOT.with(|root| {
            *root.borrow_mut() = Some(&script_task);
        });
    });

    let suuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuum = xxxxxxx.map(|x| x + 5)
                                                                          .map(|x| x / 2)
                                                                          .fold(0,
                                                                                |acc, x| acc + x);

    aaaaaaaaaaaaaaaa.map(|x| {
                        x += 1;
                        x
                    })
                    .filter(some_mod::some_filter)
}

fn floaters() {
    let z = Foo {
        field1: val1,
        field2: val2,
    };

    let x = Foo {
                field1: val1,
                field2: val2,
            }
            .method_call()
            .method_call();

    let y = if cond {
                val1
            } else {
                val2
            }
            .method_call();

    {
        match x {
            PushParam => {
                // params are 1-indexed
                stack.push(mparams[match cur.to_digit(10) {
                               Some(d) => d as usize - 1,
                               None => return Err("bad param number".to_owned()),
                           }]
                           .clone());
            }
        }
    }

    if cond {
        some();
    } else {
        none();
    }
    .bar()
    .baz();

    Foo { x: val }
        .baz(|| {
            // force multiline
        })
        .quux();

    Foo {
        y: i_am_multi_line,
        z: ok,
    }
    .baz(|| {
        // force multiline
    })
    .quux();

    a +
    match x {
        true => "yay!",
        false => "boo!",
    }
    .bar()
}

fn is_replaced_content() -> bool {
    constellat.send(ConstellationMsg::ViewportConstrained(self.id, constraints))
              .unwrap();
}

fn issue587() {
    a.b::<()>(c);

    std::mem::transmute(dl.symbol::<()>("init").unwrap())
}
