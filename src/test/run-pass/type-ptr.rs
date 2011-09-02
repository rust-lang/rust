fn f(a: *int) -> *int { ret a; }

fn g(a: *int) -> *int { let b = f(a); ret b; }

fn main(args: [istr]) { ret; }
