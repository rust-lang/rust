

iter x() -> int { }

fn f() -> bool { for each i: int  in x() { ret true; } ret false; }

fn main(args: vec[str]) { f(); }