pub unsafe fn f(xs: Vec<isize> ) {
    let _ = xs.iter().map(|_x| { unsafe fn q() { panic!(); } }).collect::<Vec<()>>();
}
