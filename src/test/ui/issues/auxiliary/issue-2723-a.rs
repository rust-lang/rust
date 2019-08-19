pub unsafe fn f(xs: Vec<isize> ) {
    xs.iter().map(|_x| { unsafe fn q() { panic!(); } }).collect::<Vec<()>>();
}
