// rustfmt-chains_overflow_last: false
// Test chain formatting without overflowing the last item.

fn main() {
    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
        .ddddddddddddddddddddddddddd();

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
        .ddddddddddddddddddddddddddd
        .eeeeeeee();

    x().y(|| match cond() {
            true => (),
            false => (),
        });

    loong_func()
        .quux(move || if true {
            1
        } else {
            2
        });

    fffffffffffffffffffffffffffffffffff(a,
                                        {
                                            SCRIPT_TASK_ROOT.with(|root| {
                                                    *root.borrow_mut() = Some(&script_task);
                                                });
                                        });

    let suuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuum = xxxxxxx.map(|x| x + 5)
        .map(|x| x / 2)
        .fold(0, |acc, x| acc + x);

    aaaaaaaaaaaaaaaa.map(|x| {
            x += 1;
            x
        })
        .filter(some_mod::some_filter)
}
