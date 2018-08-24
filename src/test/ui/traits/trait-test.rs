trait foo { fn foo(&self); }

impl isize for usize { fn foo(&self) {} } //~ ERROR trait

fn main() {}
