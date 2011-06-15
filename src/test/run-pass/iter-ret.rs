

iter x() -> int { }

fn f() -> bool { for each (int i in x()) { ret true; } ret false; }

fn main(vec[str] args) { f(); }