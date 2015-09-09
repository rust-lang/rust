// Test chain formatting.

fn main() {
    let a = b.c.d.1.foo(|x| x + 1);

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc.ddddddddddddddddddddddddddd();

    bbbbbbbbbbbbbbbbbbb.ccccccccccccccccccccccccccccccccccccc
        .ddddddddddddddddddddddddddd
        .eeeeeeee();

    x().y(|| {
           match cond() {
               true => (),
               false => (),
           }
       });

    loong_func()
        .quux(move || {
            if true {
                1
            } else {
                2
            }
        });

    let suuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuum = xxxxxxx.map(|x| x + 5)
                                                                       .map(|x| x / 2)
                                                                       .fold(0, |acc, x| acc + x);

    aaaaaaaaaaaaaaaa
        .map(|x| {
            x += 1;
            x
        })
        .filter(some_mod::some_filter)
}
