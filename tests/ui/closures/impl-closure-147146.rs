impl typeof(|| {}) {}
//~^ ERROR `typeof` is a reserved keyword but unimplemented

unsafe impl Send for typeof(|| {}) {}
//~^ ERROR `typeof` is a reserved keyword but unimplemented

fn main() {}
