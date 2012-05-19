fn test(cond: bool) {
    let v: int;
    v = 1;
    loop { } // loop never terminates, so no error is reported
    v = 2;
}

fn main() {
	// note: don't call test()... :)
}
