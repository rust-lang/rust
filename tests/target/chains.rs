// Test chain formatting.

fn main() {
    // Don't put chains on a single line if it wasn't so in source.
    let a = b.c
             .d
             .1
             .foo(|x| x + 1);

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc.ddddddddddddddddddddddddddd();

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
