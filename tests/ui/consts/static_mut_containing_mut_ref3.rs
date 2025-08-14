static mut FOO: (u8, u8) = (42, 43);

static mut BAR: () = unsafe { FOO.0 = 99; };
//~^ ERROR modifying a static's initial value

fn main() {}
