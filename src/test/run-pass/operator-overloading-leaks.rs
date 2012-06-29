// The cases commented as "Leaks" need to not leak. Issue #2581

impl methods<T: copy> for [T]/~ {
    fn -(x: [T]/&) -> [T]/~ {
        [x[0], x[0], x[0]]/~
    }

    fn foo(x: [T]/&) -> [T]/~ {
        [x[0], x[0], x[0]]/~
    }
}

impl methods<T: copy> for ~T {
    fn +(rhs: ~T) -> ~T {
        rhs
    }
}

impl methods for ~int {
    fn -(rhs: ~int) -> ~int {
        ~(*self - *rhs)
    }
}

impl methods for @int {
    fn +(rhs: @int) -> @int {
        @(*self + *rhs)
    }
}

fn main() {
    // leaks
    let mut bar = [1, 2, 3]/~;
    bar -= [3, 2, 1]/~;
    bar -= [4, 5, 6]/~;
    
    io::println(#fmt("%?", bar));

    // okay
    let mut bar = [1, 2, 3]/~;
    bar = bar.foo([3, 2, 1]/~);
    bar = bar.foo([4, 5, 6]/~);

    io::println(#fmt("%?", bar));

    // okay
    let mut bar = [1, 2, 3]/~;
    bar = bar - [3, 2, 1]/~;
    bar = bar - [4, 5, 6]/~;

    io::println(#fmt("%?", bar));

    // Leaks
    let mut bar = ~1;
    bar += ~2;
    bar += ~3;
    
    io:: println(#fmt("%?", bar));

    // Leaks
    let mut bar = ~1;
    bar -= ~2;
    bar -= ~3;
    
    io:: println(#fmt("%?", bar));

    // Leaks
    let mut bar = @1;
    bar += @2;
    bar += @3;
    
    io:: println(#fmt("%?", bar));

}
